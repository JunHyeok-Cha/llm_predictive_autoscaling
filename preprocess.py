"""
BurstGPT 데이터 전처리 파이프라인
딥러닝분석 Final Project — Predictive Autoscaling for LLM Inference

처리 순서:
  1. 데이터 로딩 (without_fails 1+2 병합, 3 별도)
  2. Timestamp → datetime 변환
  3. 실패율(failure_rate) feature 추출 (원본 with_fails 활용)
  4. 이상치 클리핑
  5. 1분 단위 집계 → 6개 feature 생성
  6. 빈 분 패턴 분석 및 처리
  7. Feature Engineering (원형 인코딩, lag, rolling)
  8. Walk-Forward (Expanding Window) Split
  9. MinMaxScaler 정규화 (fold별 독립 fit)
  10. 전처리 결과 저장
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# 0. 경로 설정
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "BurstGPT", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed")
PLOT_DIR   = os.path.join(BASE_DIR, "preprocess_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# BurstGPT_1 Timestamp=0 기준일 (논문 미공개 → 임의 설정, 요일 패턴 보존됨)
BASE_DATE = pd.Timestamp("2023-01-02")  # 월요일 시작으로 설정

# 예측 타겟 컬럼
TARGET_COL = "req_count"

# lag / rolling 윈도우 (분 단위)
LAG_WINDOWS     = [1, 60, 1440, 10080]          # 1분, 1시간, 1일, 1주
ROLLING_WINDOWS = [60, 1440]                     # 1시간, 24시간
PRED_HORIZON    = 30                             # 30분 후 예측

# Walk-Forward 설정 (분 단위)
INITIAL_TRAIN_DAYS = 30   # 최소 학습 기간 (lag_10080 × 4사이클)
VAL_DAYS           = 14   # 검증 윈도우 (2주 = 주간 주기 2사이클)
N_FOLDS            = 5    # fold 수 (step은 전체 CV 구간에서 자동 계산)
FINAL_TEST_DAYS    = 21   # 최종 holdout test (절대 학습 미사용)

print("=" * 60)
print("BurstGPT 전처리 파이프라인 시작")
print("=" * 60)


# ============================================================
# 1. 데이터 로딩
# ============================================================
print("\n[1] 데이터 로딩")

def load_burst(path, label=""):
    print(f"  로딩: {os.path.basename(path)} ...", end=" ")
    df = pd.read_csv(path)
    print(f"{len(df):,} rows")
    df["source"] = label
    return df

# without_fails: 1+2 병합 (121일 연속) → 메인 학습용
df_main = pd.concat([
    load_burst(os.path.join(DATA_DIR, "BurstGPT_without_fails_1.csv"), "main_1"),
    load_burst(os.path.join(DATA_DIR, "BurstGPT_without_fails_2.csv"), "main_2"),
], ignore_index=True)
print(f"  → 1+2 병합: {len(df_main):,} rows")

# without_fails_3: 타임스탬프를 2번 파일 직후로 붙여서 연속 결합
df_period3_raw = load_burst(
    os.path.join(DATA_DIR, "BurstGPT_without_fails_3.csv"), "period3"
)

# 파일 2 마지막 ts → 파일 3 첫 ts 사이 공백 제거 (60초 간격으로 이어붙임)
ts_offset = (df_main["Timestamp"].max()
             - df_period3_raw["Timestamp"].min()
             + 60)  # 1분 간격
df_period3_raw = df_period3_raw.copy()
df_period3_raw["Timestamp"] = df_period3_raw["Timestamp"] + ts_offset
print(f"  period3 Timestamp offset: +{ts_offset:.0f}초 "
      f"({ts_offset/86400:.1f}일) 적용")

# 전체 합치기: 1+2+3 연속 데이터
df_all = pd.concat([df_main, df_period3_raw], ignore_index=True)
print(f"  → 1+2+3 병합: {len(df_all):,} rows  "
      f"(Ts 범위: {df_all['Timestamp'].min():.0f} ~ {df_all['Timestamp'].max():.0f})")

# with_fails: 실패율 추출용 — 청크 단위로 분당 실패율만 추출 (메모리 절약)
def extract_failure_rate_chunked(paths, base_date, ts_offsets=None, chunksize=200_000):
    """
    원본(with_fails) 파일을 청크로 읽어 분당 실패율만 집계.
    전체 파일을 메모리에 올리지 않아도 됨.
    """
    # ts_offsets: 각 파일에 더할 초 단위 오프셋 (None이면 0)
    if ts_offsets is None:
        ts_offsets = [0] * len(paths)

    bucket_total = {}
    bucket_fail  = {}
    for path, ts_off in zip(paths, ts_offsets):
        print(f"    청크 처리: {os.path.basename(path)} ...", end=" ", flush=True)
        chunks = 0
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk["Timestamp"] = chunk["Timestamp"].astype(float) + ts_off
            chunk["datetime"]  = base_date + pd.to_timedelta(
                chunk["Timestamp"], unit="s")
            chunk["minute_key"] = chunk["datetime"].dt.floor("1min")
            chunk["is_fail"]    = (chunk["Response tokens"] == 0).astype(int)
            for key, grp in chunk.groupby("minute_key"):
                bucket_total[key] = bucket_total.get(key, 0) + len(grp)
                bucket_fail[key]  = bucket_fail.get(key,  0) + grp["is_fail"].sum()
            chunks += 1
        print(f"{chunks}청크 완료")

    total_s = pd.Series(bucket_total, name="total")
    fail_s  = pd.Series(bucket_fail,  name="fail")
    rate    = (fail_s / total_s.replace(0, np.nan)).fillna(0).rename("failure_rate")
    rate.index = pd.DatetimeIndex(rate.index)
    rate = rate.sort_index()
    print(f"    → 실패율 집계: {len(rate):,}분 | 평균={rate.mean()*100:.2f}% | 최대={rate.max()*100:.1f}%")
    return rate

# with_fails 파일도 동일한 ts_offset 적용
fail_rate_all = extract_failure_rate_chunked(
    paths=[
        os.path.join(DATA_DIR, "BurstGPT_1.csv"),
        os.path.join(DATA_DIR, "BurstGPT_2.csv"),
        os.path.join(DATA_DIR, "BurstGPT_3.csv"),
    ],
    base_date=BASE_DATE,
    ts_offsets=[0, 0, ts_offset],   # 3번 파일에만 동일 offset 적용
)


# ============================================================
# 2. Timestamp → datetime 변환
# ============================================================
print("\n[2] Timestamp → datetime 변환")

def convert_timestamp(df, base_date, col="Timestamp"):
    """초 단위 timestamp → datetime (base_date 기준)"""
    df = df.copy()
    df["datetime"] = base_date + pd.to_timedelta(df[col].astype(float), unit="s")
    return df

df_all = convert_timestamp(df_all, BASE_DATE)
print(f"  전체(1+2+3): {df_all['datetime'].min()} ~ {df_all['datetime'].max()}")


# ============================================================
# 3. 이상치 클리핑 (개별 행 수준, 집계 전)
# ============================================================
print("\n[3] 이상치 클리핑 (99.9th percentile)")

def clip_tokens(df, cols=["Request tokens", "Response tokens"]):
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        upper = df[col].quantile(0.999)
        before_max = df[col].max()
        df[col] = df[col].clip(upper=upper)
        print(f"  {col}: max {before_max:.0f} → {upper:.0f} (클리핑)")
    return df

df_all = clip_tokens(df_all)


# ============================================================
# 4. 실패율(failure_rate) 추출 — 분 단위
# ============================================================
print("\n[4] 실패율 feature 추출")



# ============================================================
# 5. 1분 단위 집계 → feature 생성
# ============================================================
print("\n[5] 1분 단위 집계")

def aggregate_1min(df):
    """
    개별 요청 → 1분 버킷 집계
    생성 features:
      req_count            : 분당 총 요청 수 (예측 타겟)
      gpt4_ratio           : GPT-4 요청 비율
      conv_ratio           : Conversation log 비율
      avg_req_tokens       : 평균 입력 토큰
      avg_resp_tokens      : 평균 출력 토큰
      total_token_throughput: 분당 총 토큰 처리량
    """
    df = df.copy()
    df = df.set_index("datetime")

    # 기본 집계
    agg = pd.DataFrame()
    agg["req_count"]             = df.resample("1min").size()
    agg["gpt4_count"]            = df["Model"].resample("1min").apply(
                                       lambda x: (x == "GPT-4").sum())
    agg["conv_count"]            = df["Log Type"].resample("1min").apply(
                                       lambda x: (x == "Conversation log").sum())
    agg["sum_req_tokens"]        = df["Request tokens"].resample("1min").sum()
    agg["sum_resp_tokens"]       = df["Response tokens"].resample("1min").sum()
    agg["total_token_throughput"]= df["Total tokens"].resample("1min").sum()

    # 비율 및 평균 계산 (0요청 분은 NaN → 0)
    rc = agg["req_count"].replace(0, np.nan)
    agg["gpt4_ratio"]     = (agg["gpt4_count"]  / rc).fillna(0)
    agg["conv_ratio"]     = (agg["conv_count"]   / rc).fillna(0)
    agg["avg_req_tokens"] = (agg["sum_req_tokens"] / rc).fillna(0)
    agg["avg_resp_tokens"]= (agg["sum_resp_tokens"] / rc).fillna(0)

    # 중간 컬럼 제거
    agg = agg.drop(columns=["gpt4_count", "conv_count",
                             "sum_req_tokens", "sum_resp_tokens"])
    return agg

ts_all = aggregate_1min(df_all)
print(f"  전체: {len(ts_all):,}분 | req_count mean={ts_all['req_count'].mean():.1f} "
      f"max={ts_all['req_count'].max():.0f}")


# ============================================================
# 6. 빈 분 분석 및 처리
# ============================================================
print("\n[6] 빈 분(0 요청) 분석")

def analyze_empty_minutes(ts, label=""):
    empty_mask = ts["req_count"] == 0
    empty_rate = empty_mask.mean()
    print(f"  [{label}] 빈 분 비율: {empty_rate*100:.1f}%  ({empty_mask.sum():,}분 / {len(ts):,}분)")

    # 시간대별 빈 분 분포 (새벽인지 확인)
    hour_empty = ts[empty_mask].index.hour
    print(f"  빈 분 집중 시간대 top-5: {pd.Series(hour_empty).value_counts().head(5).to_dict()}")
    return empty_rate

analyze_empty_minutes(ts_all, "전체 1+2+3")

# 실패율 병합
ts_all = ts_all.join(fail_rate_all, how="left").fillna({"failure_rate": 0})

# 빈 분 처리: is_zero_fill 플래그 추가 후 0 유지
ts_all["is_zero_fill"] = (ts_all["req_count"] == 0).astype(int)

# avg_req_tokens, avg_resp_tokens 빈 분: forward fill
for col in ["avg_req_tokens", "avg_resp_tokens"]:
    ts_all[col] = ts_all[col].replace(0, np.nan).ffill().fillna(0)

print(f"  처리: 0 유지 + is_zero_fill 플래그 | avg tokens: forward fill")


# ============================================================
# 7. Feature Engineering
# ============================================================
print("\n[7] Feature Engineering")

def add_time_features(ts):
    """시간 관련 feature 추가"""
    ts = ts.copy()
    hour    = ts.index.hour
    weekday = ts.index.dayofweek
    minute  = ts.index.minute

    # 원형 인코딩 (23시와 0시가 가깝도록)
    ts["hour_sin"]    = np.sin(2 * np.pi * hour    / 24)
    ts["hour_cos"]    = np.cos(2 * np.pi * hour    / 24)
    ts["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    ts["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    ts["minute_sin"]  = np.sin(2 * np.pi * minute  / 60)
    ts["minute_cos"]  = np.cos(2 * np.pi * minute  / 60)
    ts["is_weekend"]  = (weekday >= 5).astype(int)
    return ts

def add_lag_features(ts, col=TARGET_COL, windows=LAG_WINDOWS):
    """Lag feature 추가"""
    ts = ts.copy()
    for w in windows:
        ts[f"{col}_lag_{w}"] = ts[col].shift(w)
    return ts

def add_rolling_features(ts, col=TARGET_COL, windows=ROLLING_WINDOWS):
    """Rolling 통계 feature 추가"""
    ts = ts.copy()
    for w in windows:
        ts[f"{col}_roll_mean_{w}"] = ts[col].shift(1).rolling(w, min_periods=1).mean()
        ts[f"{col}_roll_std_{w}"]  = ts[col].shift(1).rolling(w, min_periods=1).std().fillna(0)
    return ts

def add_target(ts, col=TARGET_COL, horizon=PRED_HORIZON):
    """예측 타겟: horizon분 후 req_count"""
    ts = ts.copy()
    ts["target"] = ts[col].shift(-horizon)
    return ts

def engineer_features(ts, label=""):
    ts = add_time_features(ts)
    ts = add_lag_features(ts)
    ts = add_rolling_features(ts)
    ts = add_target(ts)
    # lag/rolling 생성으로 생긴 NaN 제거
    before = len(ts)
    ts = ts.dropna()
    print(f"  [{label}] {before:,}분 → NaN 제거 후 {len(ts):,}분")
    return ts

ts_all = engineer_features(ts_all, "전체 1+2+3")

print(f"\n  최종 feature 목록 ({len(ts_all.columns)}개):")
for col in ts_all.columns:
    print(f"    {col}")


# ============================================================
# 8. Walk-Forward (Expanding Window) Split
# ============================================================
print("\n[8] Walk-Forward Expanding Window Split")

MIN_PER_DAY = 1440  # 하루 = 1440분

def walk_forward_split(ts,
                       initial_train_days=INITIAL_TRAIN_DAYS,
                       val_days=VAL_DAYS,
                       n_folds=N_FOLDS,
                       final_test_days=FINAL_TEST_DAYS):
    """
    Walk-Forward Expanding Window CV + Final Holdout Test

    step은 전체 CV 구간을 n_folds로 균등 분할해 자동 계산 →
    데이터를 남김없이 전부 활용.

    구조:
      CV 구간 (전체 - final_test_days일):
        ├── Fold 1: Train(30일~) │ Val(14일) ← 균등 간격
        ├── Fold 2: Train(확장)  │ Val(14일)
        ├── ...
        └── Fold N: Train(확장)  │ Val(14일) ← CV 구간 끝에 닿음
      Final Test: 마지막 final_test_days일 (절대 학습 미사용)
    """
    n_test_min = final_test_days * MIN_PER_DAY
    cv_data    = ts.iloc[:-n_test_min]
    test_set   = ts.iloc[-n_test_min:]

    n_val_min  = val_days         * MIN_PER_DAY
    n_init_min = initial_train_days * MIN_PER_DAY

    # step: CV 구간을 n_folds로 균등 분할 (전체 데이터 낭비 없음)
    # 마지막 fold의 val_end = len(cv_data) 가 되도록 step 역산
    # val_end_k = n_init_min + k * step + n_val_min
    # k=n_folds 일 때 val_end = len(cv_data) →
    # step = (len(cv_data) - n_init_min - n_val_min) / n_folds
    available   = len(cv_data) - n_init_min - n_val_min
    n_step_min  = available // n_folds
    print(f"  자동 step: {n_step_min}분 = {n_step_min/MIN_PER_DAY:.1f}일 "
          f"(CV {len(cv_data)//MIN_PER_DAY}일 → {n_folds}등분)")

    folds = []
    for k in range(n_folds):
        val_end   = n_init_min + (k + 1) * n_step_min + n_val_min
        val_start = val_end - n_val_min
        train_end = val_start

        if val_end > len(cv_data):
            val_end   = len(cv_data)
            val_start = val_end - n_val_min
            train_end = val_start

        train_fold = cv_data.iloc[:train_end]
        val_fold   = cv_data.iloc[val_start:val_end]
        folds.append((train_fold, val_fold))

        print(f"  Fold {k+1}: "
              f"Train {len(train_fold)//MIN_PER_DAY}일 ({len(train_fold):,}분) "
              f"({train_fold.index[0].date()} ~ {train_fold.index[-1].date()}) │ "
              f"Val {len(val_fold)//MIN_PER_DAY}일 ({len(val_fold):,}분) "
              f"({val_fold.index[0].date()} ~ {val_fold.index[-1].date()})")

    print(f"  Final Test: {len(test_set):,}분 "
          f"({test_set.index[0].date()} ~ {test_set.index[-1].date()})")
    return folds, test_set

folds, final_test = walk_forward_split(ts_all)


# ============================================================
# 9. MinMaxScaler 정규화 (fold별 독립 fit)
# ============================================================
print("\n[9] 정규화 (fold별 독립 MinMaxScaler)")

# 정규화 대상: 수치형 컬럼 (원형 인코딩·플래그 제외)
exclude_cols = ["is_zero_fill", "is_weekend",
                "hour_sin", "hour_cos",
                "weekday_sin", "weekday_cos",
                "minute_sin", "minute_cos"]
scale_cols = [c for c in ts_all.columns if c not in exclude_cols]

def scale_fold(train_f, val_f, cols=scale_cols):
    """각 fold의 train으로 fit → val에 transform. leakage 없음."""
    scaler_f = MinMaxScaler()
    tr = train_f.copy()
    vl = val_f.copy()
    tr[cols] = scaler_f.fit_transform(train_f[cols])
    vl[cols] = scaler_f.transform(val_f[cols])
    return tr, vl, scaler_f

scaled_folds = []
scalers      = []
for k, (tr, vl) in enumerate(folds):
    tr_s, vl_s, sc = scale_fold(tr, vl)
    scaled_folds.append((tr_s, vl_s))
    scalers.append(sc)
    print(f"  Fold {k+1}: scaler fit on {len(tr):,}분 → val transform {len(vl):,}분")

# Final test: 마지막 fold의 scaler로 transform (가장 많은 데이터로 학습된 scaler)
last_scaler = scalers[-1]
final_test_scaled = final_test.copy()
final_test_scaled[scale_cols] = last_scaler.transform(final_test[scale_cols])

print(f"\n  정규화 대상 컬럼: {len(scale_cols)}개")
print(f"  Final test: 마지막 fold scaler (fold {len(folds)}) 사용")


# ============================================================
# 10. 저장
# ============================================================
print("\n[10] 전처리 결과 저장")
import json

# fold별 CSV 저장 (정규화 + 원본)
for k, ((tr_s, vl_s), (tr_r, vl_r)) in enumerate(
        zip(scaled_folds, folds), start=1):
    tr_s.to_csv(os.path.join(OUTPUT_DIR, f"fold{k}_train.csv"))
    vl_s.to_csv(os.path.join(OUTPUT_DIR, f"fold{k}_val.csv"))
    tr_r.to_csv(os.path.join(OUTPUT_DIR, f"fold{k}_train_raw.csv"))
    vl_r.to_csv(os.path.join(OUTPUT_DIR, f"fold{k}_val_raw.csv"))

# Final test & OOP
final_test_scaled.to_csv(os.path.join(OUTPUT_DIR, "final_test.csv"))
final_test.to_csv(os.path.join(OUTPUT_DIR,         "final_test_raw.csv"))

# Scaler 저장 (fold별 + 마지막 scaler)
scaler_data = {
    "scalers"    : scalers,       # list of MinMaxScaler (fold 1~N)
    "last_scaler": last_scaler,   # final_test에 사용
    "scale_cols" : scale_cols,
}
with open(os.path.join(OUTPUT_DIR, "scalers.pkl"), "wb") as f:
    pickle.dump(scaler_data, f)

# 메타 정보
fold_summary = []
for k, (tr_r, vl_r) in enumerate(folds, start=1):
    fold_summary.append({
        "fold"        : k,
        "train_size"  : len(tr_r),
        "val_size"    : len(vl_r),
        "train_start" : str(tr_r.index[0].date()),
        "train_end"   : str(tr_r.index[-1].date()),
        "val_start"   : str(vl_r.index[0].date()),
        "val_end"     : str(vl_r.index[-1].date()),
    })

meta = {
    "target_col"        : TARGET_COL,
    "pred_horizon"      : PRED_HORIZON,
    "lag_windows"       : LAG_WINDOWS,
    "rolling_windows"   : ROLLING_WINDOWS,
    "feature_cols"      : [c for c in scaled_folds[0][0].columns if c != "target"],
    "n_features"        : len(scaled_folds[0][0].columns) - 1,
    "n_folds"           : len(folds),
    "folds"             : fold_summary,
    "final_test_size"   : len(final_test),
    "base_date"         : str(BASE_DATE),
    "split_config": {
        "initial_train_days": INITIAL_TRAIN_DAYS,
        "val_days"          : VAL_DAYS,
        "step_days"         : "auto",  # walk_forward_split 내부에서 동적 계산
        "final_test_days"   : FINAL_TEST_DAYS,
    }
}
with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"  저장 위치: {OUTPUT_DIR}")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fsize = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"    {fname:30s} {fsize/1024/1024:.1f} MB")


# ============================================================
# 11. 전처리 검증 시각화
# ============================================================
print("\n[11] 검증 시각화 생성")

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.3})

# ── 그림 1: Walk-Forward fold 구조 시각화 ───────────────
FOLD_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
fig, axes = plt.subplots(len(folds) + 1, 1,
                          figsize=(15, 2.2 * (len(folds) + 1)),
                          sharex=True)

# 전체 시계열 (배경)
ts_hourly = ts_all["req_count"].resample("1h").sum()

for k, (tr_r, vl_r) in enumerate(folds):
    ax = axes[k]
    ax.fill_between(ts_hourly.index, ts_hourly.values,
                    alpha=0.08, color="gray")
    # 학습 구간
    tr_h = tr_r["req_count"].resample("1h").sum()
    ax.fill_between(tr_h.index, tr_h.values,
                    alpha=0.5, color=FOLD_COLORS[k], label=f"Train")
    ax.plot(tr_h.index, tr_h.values,
            color=FOLD_COLORS[k], linewidth=0.6)
    # 검증 구간
    vl_h = vl_r["req_count"].resample("1h").sum()
    ax.fill_between(vl_h.index, vl_h.values,
                    alpha=0.9, color="orange", label="Val")
    ax.plot(vl_h.index, vl_h.values, color="orange", linewidth=0.8)

    ax.set_ylabel("Req/h", fontsize=8)
    ax.set_title(
        f"Fold {k+1}  │  Train {len(tr_r)//1440}일 "
        f"({tr_r.index[0].date()}~{tr_r.index[-1].date()})  │  "
        f"Val {len(vl_r)//1440}일 "
        f"({vl_r.index[0].date()}~{vl_r.index[-1].date()})",
        fontsize=9)
    ax.legend(fontsize=8, loc="upper left")

# Final test
ax = axes[-1]
ax.fill_between(ts_hourly.index, ts_hourly.values, alpha=0.08, color="gray")
ft_h = final_test["req_count"].resample("1h").sum()
ax.fill_between(ft_h.index, ft_h.values, alpha=0.9, color="#2ecc71", label="Final Test")
ax.plot(ft_h.index, ft_h.values, color="#2ecc71", linewidth=0.8)
ax.set_ylabel("Req/h", fontsize=8)
ax.set_title(f"Final Holdout Test  │  {len(final_test)//1440}일 "
             f"({final_test.index[0].date()}~{final_test.index[-1].date()})",
             fontsize=9)
ax.legend(fontsize=8, loc="upper left")

fig.suptitle("Walk-Forward Expanding Window CV — BurstGPT 1+2+3 (전체 데이터)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "01_walkforward_split.png"), bbox_inches="tight")
plt.close()
print("  saved: 01_walkforward_split.png")

# ── 그림 2: 빈 분 시간대 분포 ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
zero_hours = ts_all[ts_all["is_zero_fill"] == 1].index.hour
pd.Series(zero_hours).value_counts().sort_index().plot.bar(ax=ax, color="#C44E52", alpha=0.8)
ax.set_title("빈 분(0-request) 시간대 분포 — 새벽 저부하 시간 집중 확인", fontsize=11)
ax.set_xlabel("Hour of Day (UTC)")
ax.set_ylabel("빈 분 수")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "02_empty_minute_hours.png"), bbox_inches="tight")
plt.close()
print("  saved: 02_empty_minute_hours.png")

# ── 그림 3: req_count 분포 (fold별 train/val + final test + oop) ──
last_train_r, last_val_r = folds[-1]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (label, df_s, color) in zip(axes, [
    (f"Fold{len(folds)} Train", last_train_r, "#4C72B0"),
    (f"Fold{len(folds)} Val",   last_val_r,   "#DD8452"),
    ("Final Test",              final_test,    "#55A868"),
]):
    df_s["req_count"].clip(upper=df_s["req_count"].quantile(0.999)).hist(
        ax=ax, bins=60, color=color, alpha=0.8, edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(f"{label}\nmean={df_s['req_count'].mean():.1f}")
    ax.set_xlabel("req_count")
    ax.set_ylabel("Count (log)")
fig.suptitle("req_count Distribution — Last Fold Train / Val / Final Test",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "03_reqcount_dist.png"), bbox_inches="tight")
plt.close()
print("  saved: 03_reqcount_dist.png")

# ── 그림 4: 실패율 시계열 ────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 3))
fail_hourly = ts_all["failure_rate"].resample("1h").mean()
fail_hourly.plot(ax=ax, color="#C44E52", linewidth=0.7, alpha=0.9)
ax.set_title("Hourly Average Failure Rate (main 1+2)", fontsize=11)
ax.set_ylabel("Failure Rate")
ax.set_xlabel("Time")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "04_failure_rate.png"), bbox_inches="tight")
plt.close()
print("  saved: 04_failure_rate.png")

# ── 그림 5: feature 상관관계 히트맵 (train) ─────────────
corr_cols = [TARGET_COL, "gpt4_ratio", "conv_ratio",
             "avg_req_tokens", "avg_resp_tokens",
             "total_token_throughput", "failure_rate",
             f"{TARGET_COL}_lag_1", f"{TARGET_COL}_lag_1440",
             f"{TARGET_COL}_roll_mean_60", "target"]
import seaborn as sns
fig, ax = plt.subplots(figsize=(11, 9))
# fold1 train (가장 짧은 train) 기준 상관관계
corr = folds[0][0][corr_cols].corr()
sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.3, annot_kws={"size": 8})
ax.set_title("Feature Correlation Matrix (Train set)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "05_feature_correlation.png"), bbox_inches="tight")
plt.close()
print("  saved: 05_feature_correlation.png")

# ── 그림 6: lag feature 유효성 확인 ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
lag_pairs = [
    (f"{TARGET_COL}_lag_1",    "lag 1분",   "#4C72B0"),
    (f"{TARGET_COL}_lag_60",   "lag 1시간", "#DD8452"),
    (f"{TARGET_COL}_lag_1440", "lag 1일",   "#55A868"),
    (f"{TARGET_COL}_lag_10080","lag 1주",   "#C44E52"),
]
sample = folds[-1][0].sample(min(3000, len(folds[-1][0])), random_state=42)
for ax, (lag_col, lag_label, color) in zip(axes.flat, lag_pairs):
    ax.scatter(sample[lag_col], sample["target"],
               alpha=0.15, s=3, color=color)
    corr_val = sample[[lag_col, "target"]].corr().iloc[0, 1]
    ax.set_title(f"{lag_label}  (r={corr_val:.3f})")
    ax.set_xlabel(lag_col)
    ax.set_ylabel("target (30분 후)")
fig.suptitle("Lag Feature vs Target Correlation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "06_lag_vs_target.png"), bbox_inches="tight")
plt.close()
print("  saved: 06_lag_vs_target.png")

# ============================================================
# 최종 요약
# ============================================================
print("\n" + "=" * 60)
print("전처리 완료 요약")
print("=" * 60)
print(f"  Feature 수       : {meta['n_features']}개")
print(f"  예측 horizon     : {PRED_HORIZON}분 후")
print(f"  Walk-forward fold: {meta['n_folds']}개")
for f in meta["folds"]:
    print(f"    Fold {f['fold']}: Train {f['train_size']:,}분  Val {f['val_size']:,}분")
print(f"  Final Test       : {meta['final_test_size']:,}분")
print(f"\n  저장 → {OUTPUT_DIR}")
print(f"  플롯 → {PLOT_DIR}")
