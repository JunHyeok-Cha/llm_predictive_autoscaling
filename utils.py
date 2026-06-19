"""
utils.py — 공통 유틸리티 (Keras/TensorFlow)
  - GPU 설정 (메모리 증가 모드)
  - 데이터 로드
  - 슬라이딩 윈도우 배열 생성
  - 평가 지표 (MAE, RMSE, MAPE)
  - tqdm 진행바 Keras 콜백
  - 결과 저장 / 시각화
"""
import gc
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── GPU 설정 (import 시점에 즉시 실행) ───────────────────────────────────────
import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"  🟢 GPU 사용: {[g.name for g in gpus]}")
        # mixed_float16은 Tensor Core 보유 GPU (Ampere CC≥8.0) 에서만 효과적.
        # GTX 1660 SUPER (CC 7.5) 등 Tensor Core 없는 GPU에서 활성화 시
        # XLA autotuner가 cuBLAS FP16 알고리즘을 찾지 못해 InternalError 발생.
        details  = tf.config.experimental.get_device_details(gpus[0])
        cc_major = details.get("compute_capability", (0, 0))[0]
        if cc_major >= 8:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            print(f"  🟢 Mixed Precision (FP16) 활성화 (CC {cc_major}.x Ampere+)")
        else:
            print(f"  🟡 Mixed Precision 스킵 (CC {cc_major}.x — Tensor Core 없음, float32 사용)")
    else:
        print("  🟡 GPU 없음 — CPU로 실행")
    return bool(gpus)

HAS_GPU = setup_gpu()

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PREP_DIR    = os.path.join(BASE_DIR, "preprocessed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── meta.json 로드 ────────────────────────────────────────────────────────────
with open(os.path.join(PREP_DIR, "meta.json")) as f:
    META = json.load(f)

FEATURE_COLS  = META["feature_cols"]   # 23개
TARGET_COL    = META["target_col"]     # "req_count"
N_FOLDS       = META["n_folds"]        # 5
N_FEATURES    = META["n_features"]     # 23
PRED_HORIZON  = META["pred_horizon"]   # 30

# TFT용: 미래에도 알 수 있는 피처 (시간 인코딩·요일)
KNOWN_FUTURE_COLS = [
    "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos",
    "minute_sin", "minute_cos",
    "is_weekend",
]
PAST_ONLY_COLS = [c for c in FEATURE_COLS if c not in KNOWN_FUTURE_COLS]
N_PAST   = len(PAST_ONLY_COLS)
N_KNOWN  = len(KNOWN_FUTURE_COLS)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_fold(fold: int):
    tr = pd.read_csv(os.path.join(PREP_DIR, f"fold{fold}_train.csv"),
                     index_col=0, parse_dates=True)
    vl = pd.read_csv(os.path.join(PREP_DIR, f"fold{fold}_val.csv"),
                     index_col=0, parse_dates=True)
    return tr, vl

def load_test():
    return pd.read_csv(os.path.join(PREP_DIR, "final_test.csv"),
                       index_col=0, parse_dates=True)

def load_scalers():
    with open(os.path.join(PREP_DIR, "scalers.pkl"), "rb") as f:
        return pickle.load(f)


# ── 슬라이딩 윈도우 배열 생성 ─────────────────────────────────────────────────
def make_sequences(df: pd.DataFrame, seq_len: int):
    """
    검증/테스트용 — 데이터가 작으므로 numpy 배열 그대로 반환.
    X: (N, seq_len, n_features)   y: (N,)
    """
    X_arr = df[FEATURE_COLS].values.astype(np.float32)
    y_arr = df["target"].values.astype(np.float32)

    valid = np.where(
        ~(np.isnan(y_arr[seq_len:]) | np.isnan(X_arr[seq_len:]).any(axis=1))
    )[0] + seq_len

    # 단일 fancy-index로 한 번에 할당 (중간 리스트 없음)
    win_idx = valid[:, None] + np.arange(-seq_len, 0)   # (N, seq_len)
    return X_arr[win_idx], y_arr[valid]


def make_train_dataset(df: pd.DataFrame, seq_len: int, batch_size: int):
    """
    훈련용 — tf.data 제너레이터로 배치를 on-the-fly 생성.
    전체 (N, seq_len, n_features) 배열을 메모리에 올리지 않음.
    peak 메모리: raw CSV 크기(~12 MB) + 배치 1개(~4 MB).
    """
    X_arr = df[FEATURE_COLS].values.astype(np.float32)
    y_arr = df["target"].values.astype(np.float32)

    valid = np.where(
        ~(np.isnan(y_arr[seq_len:]) | np.isnan(X_arr[seq_len:]).any(axis=1))
    )[0] + seq_len
    n_samples = len(valid)

    def _gen():
        idx = valid.copy()
        np.random.shuffle(idx)
        for i in idx:
            yield X_arr[i - seq_len:i], y_arr[i]

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, N_FEATURES), dtype=tf.float32),
            tf.TensorSpec(shape=(),                   dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), n_samples


def make_tft_sequences(df: pd.DataFrame, seq_len: int):
    """
    TFT용.
    past_X:   (N, seq_len, n_past)
    future_X: (N, pred_horizon, n_known)
    y:        (N,)
    """
    past_arr   = df[PAST_ONLY_COLS].values.astype(np.float32)
    future_arr = df[KNOWN_FUTURE_COLS].values.astype(np.float32)
    y_arr      = df["target"].values.astype(np.float32)

    # 벡터화 — Python 리스트 누적(np.array 변환 시 2배 전이메모리) 없이 단일 할당
    idx_range = np.arange(seq_len, len(y_arr) - PRED_HORIZON)
    valid = idx_range[
        ~(np.isnan(y_arr[idx_range]) | np.isnan(past_arr[idx_range]).any(axis=1))
    ]
    past_win = valid[:, None] + np.arange(-seq_len, 0)            # (N, seq_len)
    fut_win  = valid[:, None] + np.arange(0, PRED_HORIZON)        # (N, horizon)
    return past_arr[past_win], future_arr[fut_win], y_arr[valid]


def make_tft_train_dataset(df: pd.DataFrame, seq_len: int, batch_size: int):
    """
    TFT 훈련용 — tf.data 제너레이터로 (past, future), y 배치를 on-the-fly 생성.
    make_tft_sequences처럼 전체 (N, seq_len, n_past) dense 배열을 메모리에 올리지
    않으므로 OOM(시스템 RAM) 회피. peak 메모리: raw CSV(~12 MB) + 배치 1개.
    """
    past_arr   = df[PAST_ONLY_COLS].values.astype(np.float32)
    future_arr = df[KNOWN_FUTURE_COLS].values.astype(np.float32)
    y_arr      = df["target"].values.astype(np.float32)

    idx_range = np.arange(seq_len, len(y_arr) - PRED_HORIZON)
    valid = idx_range[
        ~(np.isnan(y_arr[idx_range]) | np.isnan(past_arr[idx_range]).any(axis=1))
    ]
    n_samples = len(valid)

    def _gen():
        idx = valid.copy()
        np.random.shuffle(idx)
        for i in idx:
            yield ((past_arr[i - seq_len:i], future_arr[i:i + PRED_HORIZON]),
                   y_arr[i])

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            (tf.TensorSpec(shape=(seq_len,      N_PAST),  dtype=tf.float32),
             tf.TensorSpec(shape=(PRED_HORIZON, N_KNOWN), dtype=tf.float32)),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), n_samples


# ── 평가 지표 ─────────────────────────────────────────────────────────────────
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true, y_pred):
    # SMAPE: 분모가 (|y_true|+|y_pred|)/2 이므로 0값에서도 발산하지 않음.
    # 범위 0~200%. 두 값 모두 0인 샘플은 제외.
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask  = denom > 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def compute_metrics(y_true, y_pred):
    return {"MAE":   mae(y_true, y_pred),
            "RMSE":  rmse(y_true, y_pred),
            "SMAPE": smape(y_true, y_pred)}


# ── 스케줄링 인지 평가 지표 (Phase1/Phase2 공통) ─────────────────────────────
def violation_rate(y_true, decision_pred):
    """용량 결정값이 실제보다 작아 SLA 위반이 발생한 비율 (0~1)."""
    return float(np.mean(decision_pred < y_true))

def avg_overprovision(y_true, decision_pred):
    """평균 과대프로비저닝량 (decision_pred - y_true)+."""
    return float(np.mean(np.maximum(decision_pred - y_true, 0)))

def avg_underprovision(y_true, decision_pred):
    """평균 과소프로비저닝량 (y_true - decision_pred)+."""
    return float(np.mean(np.maximum(y_true - decision_pred, 0)))

def peak_violation_rate(y_true, decision_pred, q=0.9):
    """고부하(스파이크) 구간 한정 SLA 위반율.

    전역 violation_rate은 타겟의 80%가 0 근처인 calm 구간에 지배되어, 넓은
    안전마진(P90)만 깔아도 낮게 나온다(과소평가). 정작 용량이 중요한 부하 피크
    에서의 위반을 따로 측정하기 위해, 평가셋 y_true 상위 (1-q) 분위(기본 상위
    10%)만 골라 위반율을 계산한다. 해당 구간 샘플이 없으면 NaN."""
    y_true        = np.asarray(y_true, dtype=np.float64)
    decision_pred = np.asarray(decision_pred, dtype=np.float64)
    thr  = np.quantile(y_true, q)
    mask = y_true >= thr
    if not mask.any():
        return float("nan")
    return float(np.mean(decision_pred[mask] < y_true[mask]))

def sal_eval_score(y_true, pred, under_ratio, violation_penalty, over_coef=1.0):
    """
    학습 손실과 무관하게 모든 결과를 동일 기준(SAL_SCENARIOS['balanced'])으로
    비교하기 위한 평가용 SAL 스코어. 작을수록 좋음.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    pred   = np.asarray(pred,   dtype=np.float64)
    err   = y_true - pred
    under = (err > 0).astype(np.float64) * (under_ratio * err + violation_penalty)
    over  = (err < 0).astype(np.float64) * (over_coef * (-err))
    return float(np.mean(under + over))


# ── 분위수(Pinball) 손실 — BiLSTM / iTransformer / TFT 공통 ──────────────────
QUANTILES = [0.1, 0.5, 0.9]   # P10, P50, P90

def quantile_loss(quantiles):
    """
    각 분위수 τ에 대한 pinball loss 합산.
    y_true: (batch,)  y_pred: (batch, n_q)
    τ=0.9: 과소예측(P90 < 실제)에 큰 패널티 → Pod 용량 결정 안전 마진.
    """
    q = tf.constant(quantiles, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.expand_dims(tf.cast(y_true, tf.float32), -1)   # (batch, 1)
        err    = y_true - y_pred                                    # (batch, n_q)
        loss   = tf.maximum(q * err, (q - 1.0) * err)
        return tf.reduce_mean(loss)

    loss_fn.__name__ = "quantile_loss"
    return loss_fn


# ── SAL (Scheduling-Aware Loss) — asymmetric_mae 일반화 ──────────────────────
def sal_loss(under_ratio: float, violation_penalty: float, over_coef: float = 1.0):
    """
    HRS 논문(arXiv 2508.12839)의 SAL 일반화 버전.
      under_ratio       : 과소예측 1단위당 패널티 배수 (R, 매출손실 계수)
      violation_penalty : 과소예측 발생 시 추가되는 고정 SLA 위반 패널티 (P)
      over_coef         : 과대예측 1단위당 패널티 배수 (C, 과잉프로비저닝 비용)
    point 출력(Dense(1)) 전용 — y_true/y_pred 를 1-D로 정렬해 broadcasting 사고 방지.
    """
    R, P, C = float(under_ratio), float(violation_penalty), float(over_coef)

    def loss_fn(y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), [-1])
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
        err = y_true - y_pred
        under_mask = tf.cast(err > 0, tf.float32)
        over_mask  = tf.cast(err < 0, tf.float32)
        under_loss = under_mask * (R * err + P)
        over_loss  = over_mask  * (C * (-err))
        return tf.reduce_mean(under_loss + over_loss)

    loss_fn.__name__ = f"sal_loss_r{R}_p{P}_c{C}"
    return loss_fn


# ── U/O 비율 시나리오 ────────────────────────────────────────────────────────
# violation_penalty 보정: 타겟(req_count)은 MinMaxScaler로 [0,1] 정규화됨
# (req_count/1104). 정규화 타겟 std ≈ 0.068. 논문 잠정값(0.05/0.1)은 std의
# 73%/147%로 너무 커서 선형항을 압도 → 학습 붕괴. 따라서 std의 약 5%/10%로
# 재조정: balanced≈0.0034, aggressive≈0.0068. conservative는 0.
_TARGET_STD = 0.068   # fold5 train 'target' 표준편차 (정규화 스케일)
SAL_SCENARIOS = {
    "conservative": {"under_ratio": 2.0,  "violation_penalty": 0.0,    "over_coef": 1.0},
    "balanced":     {"under_ratio": 5.0,  "violation_penalty": 0.0034, "over_coef": 1.0},  # ~5% std
    "aggressive":   {"under_ratio": 10.0, "violation_penalty": 0.0068, "over_coef": 1.0},  # ~10% std
}
print("  ⚙️  SAL_SCENARIOS (violation_penalty는 정규화 타겟 std≈"
      f"{_TARGET_STD:.3f} 기준 재조정):")
for _name, _p in SAL_SCENARIOS.items():
    print(f"      {_name:<13} under_ratio={_p['under_ratio']:>4}  "
          f"violation_penalty={_p['violation_penalty']:.4f}  "
          f"over_coef={_p['over_coef']}")


# ── 예측값 → 확장 평가 지표 (Phase1 quantile / Phase2 point 공통) ───────────
def evaluate_predictions(y_true, pred, output_mode, eval_sal_params):
    """
    output_mode="quantile": pred (N,3)=P10/P50/P90.
        MAE/RMSE/SMAPE는 P50, violation류·sal_eval_score는 P90을 decision_pred로,
        추가로 band_width=mean(P90-P10) 기록.
    output_mode="point": pred (N,)=단일 point 출력으로 모든 지표 계산.
    sal_eval_score는 항상 eval_sal_params(기본 balanced)로 고정 계산.
    """
    y_true = np.asarray(y_true).reshape(-1)
    if output_mode == "quantile":
        pred = np.asarray(pred)
        p10, p50, p90 = pred[:, 0], pred[:, 1], pred[:, 2]
        point_pred    = p50
        decision_pred = p90
    else:
        decision_pred = np.asarray(pred).reshape(-1)
        point_pred    = decision_pred

    m = compute_metrics(y_true, point_pred)
    m["violation_rate"]     = violation_rate(y_true, decision_pred)
    m["violation_rate_peak"] = peak_violation_rate(y_true, decision_pred)
    m["avg_overprovision"]  = avg_overprovision(y_true, decision_pred)
    m["avg_underprovision"] = avg_underprovision(y_true, decision_pred)
    m["sal_eval_score"]     = sal_eval_score(y_true, decision_pred, **eval_sal_params)
    if output_mode == "quantile":
        m["band_width"] = float(np.mean(p90 - p10))
    return m


# ── tqdm 진행바 Keras 콜백 ───────────────────────────────────────────────────
from tensorflow import keras as _keras

class TqdmCallback(_keras.callbacks.Callback):
    """
    epoch 단위 tqdm 진행바.
    매 epoch 끝마다 train_loss / val_loss / lr 을 바에 표시.
    """
    def __init__(self, total_epochs: int, fold: int, model_name: str):
        super().__init__()
        from tqdm import tqdm
        self.pbar = tqdm(
            total=total_epochs,
            desc=f"  [{model_name}] Fold {fold}",
            unit="epoch",
            dynamic_ncols=True,
            colour="cyan",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(self.model.optimizer.learning_rate)
        self.pbar.set_postfix(
            train=f"{logs.get('loss', 0):.4f}",
            val=f"{logs.get('val_loss', 0):.4f}",
            lr=f"{lr:.2e}",
        )
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()


# ── 결과 저장 ─────────────────────────────────────────────────────────────────
def save_loss_curve(history, fold: int, file_tag: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(history.history["loss"],     label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title(f"{file_tag} — Fold {fold} Loss Curve")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{file_tag}_fold{fold}_loss.png")
    plt.savefig(path, dpi=120); plt.close()

def save_pred_plot(y_true, y_pred, file_tag: str, n: int = 2000):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[:n], label="Actual",    alpha=0.8, lw=0.9)
    ax.plot(y_pred[:n], label="Predicted", alpha=0.8, lw=0.9)
    ax.set_title(f"{file_tag} — Final Test Predictions (first {n} steps)")
    ax.set_xlabel("Time step (min)"); ax.set_ylabel("req_count (normalized)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{file_tag}_test_pred.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  예측 그래프 저장: {path}")

def result_exists(model_name: str, phase_tag: str) -> bool:
    """해당 (모델, phase)의 결과 JSON이 이미 있으면 True — resume(중단 후 이어하기)용."""
    return os.path.exists(
        os.path.join(RESULTS_DIR, f"{model_name}_{phase_tag}_results.json")
    )

def save_results(fold_results: list, test_metrics: dict, model_name: str, phase_tag: str):
    # fold_results의 모든 수치 키(fold 제외)에 대해 CV 평균 계산
    metric_keys = [k for k in fold_results[0] if k != "fold"]
    cv_mean = {k: float(np.mean([r[k] for r in fold_results])) for k in metric_keys}
    summary = {
        "model": model_name,
        "phase_tag": phase_tag,
        "fold_val_metrics": fold_results,
        "cv_mean": cv_mean,
        "test_metrics": test_metrics,
    }
    path = os.path.join(RESULTS_DIR, f"{model_name}_{phase_tag}_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  결과 저장: {path}")
    return summary


# ── Walk-Forward CV 공통 실행기 (BiLSTM / iTransformer 용) ───────────────────
def run_walk_forward(
    build_fn,            # build_fn(output_mode) → keras.Model (fold마다 새 인스턴스)
    seq_len: int,
    model_name: str,
    output_mode: str,     # "point" | "quantile"
    loss_fn,              # keras 호환 손실 함수
    phase_tag: str,        # "quantile" | "sal_conservative" | "sal_balanced" | "sal_aggressive"
    eval_sal_params=None,  # 기본값 SAL_SCENARIOS["balanced"] (공통 평가 기준)
    batch_size: int = 256,
    max_epochs: int = 60,
    patience:   int = 8,
):
    """
    build_fn(output_mode): 호출할 때마다 새 Keras 모델을 반환.
      output_mode="point"    → Dense(1),  predict 결과를 reshape(-1)로 decision_pred.
      output_mode="quantile" → Dense(3),  predict (N,3)에서 P50/P90 분리.
    """
    from tensorflow import keras
    from tqdm import tqdm

    # Phase 경계 메모리 정리 (이전 phase의 그래프/세션 해제 → OOM 완화)
    keras.backend.clear_session()
    gc.collect()

    if eval_sal_params is None:
        eval_sal_params = SAL_SCENARIOS["balanced"]

    file_tag = f"{model_name}_{phase_tag}"
    disp     = f"{model_name}-{phase_tag}"

    fold_results = []
    last_model   = None

    # ── 전체 진행 표시 (fold 단위) ─────────────────────────────────────────
    fold_bar = tqdm(
        range(1, N_FOLDS + 1),
        desc=f"[{disp}] 전체 진행",
        unit="fold",
        position=0,
        colour="green",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} fold [{elapsed}] {postfix}",
    )

    for fold in fold_bar:
        fold_bar.set_postfix(현재=f"Fold {fold}")
        print(f"\n{'─'*55}")
        print(f"  [{disp.upper()}]  Fold {fold} / {N_FOLDS}")
        print(f"{'─'*55}")

        # 이전 fold 모델 + TF 그래프 완전 해제
        # clear_session() 없으면 옵티마이저·레이어 그래프가 fold마다 누적됨
        if last_model is not None:
            del last_model
            last_model = None
            keras.backend.clear_session()
            gc.collect()

        tr_df, vl_df = load_fold(fold)

        # 훈련: 제너레이터 기반 — 전체 배열 불필요 (~12 MB raw data만 사용)
        tr_ds, n_tr  = make_train_dataset(tr_df, seq_len, batch_size)
        # 검증: numpy — val 세트는 작아서 허용 (~300 MB)
        X_vl, y_vl   = make_sequences(vl_df, seq_len)
        print(f"  데이터  train_samples={n_tr}  val={X_vl.shape}")

        del tr_df, vl_df   # raw CSV 즉시 해제
        gc.collect()

        model = build_fn(output_mode)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=loss_fn,
            jit_compile=False,   # XLA 끔: Turing(GTX 16xx) cuBLAS gemm autotuner 실패 회피
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=0,
            ),  # patience는 호출부(bilstm.py 등)에서 주입 — 현재 12
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0,
            ),
            TqdmCallback(
                total_epochs=max_epochs,
                fold=fold,
                model_name=disp,
            ),
        ]

        history = model.fit(
            tr_ds,
            validation_data=(X_vl, y_vl),
            epochs=max_epochs,
            callbacks=callbacks,
            verbose=0,
        )

        vl_preds = model.predict(X_vl, verbose=0)
        if output_mode == "point":
            vl_preds = vl_preds.reshape(-1)
        metrics  = evaluate_predictions(y_vl, vl_preds, output_mode, eval_sal_params)
        metrics["fold"] = fold
        fold_results.append(metrics)

        print(f"\n  ✅ Fold {fold} 결과 │ "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"SMAPE={metrics['SMAPE']:.2f}%  "
              f"ViolRate={metrics['violation_rate']:.3f}  "
              f"ViolPeak={metrics['violation_rate_peak']:.3f}  "
              f"SAL={metrics['sal_eval_score']:.4f}")

        save_loss_curve(history, fold, file_tag)
        last_model = model

        del tr_ds, X_vl, y_vl
        gc.collect()

    fold_bar.close()

    # ── 최종 테스트 ──────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  [{disp.upper()}]  Final Test")
    print(f"{'═'*55}")
    test_df    = load_test()
    X_te, y_te = make_sequences(test_df, seq_len)

    te_preds = last_model.predict(
        X_te, verbose=0,
        callbacks=[keras.callbacks.LambdaCallback(
            on_predict_begin=lambda _: print("  예측 중...", end="", flush=True),
            on_predict_end=lambda _:   print(" 완료"),
        )],
    )
    if output_mode == "point":
        te_preds = te_preds.reshape(-1)

    test_metrics = evaluate_predictions(y_te, te_preds, output_mode, eval_sal_params)
    print(f"  🏁 Test 결과 │ "
          f"MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}  "
          f"SMAPE={test_metrics['SMAPE']:.2f}%  "
          f"ViolRate={test_metrics['violation_rate']:.3f}  "
          f"ViolPeak={test_metrics['violation_rate_peak']:.3f}  "
          f"SAL={test_metrics['sal_eval_score']:.4f}")

    # CV 평균 요약
    cv_mae   = float(np.mean([r["MAE"]   for r in fold_results]))
    cv_smape = float(np.mean([r["SMAPE"] for r in fold_results]))
    print(f"\n  📊 CV 평균 │ MAE={cv_mae:.4f}  SMAPE={cv_smape:.2f}%")

    # 예측 플롯: quantile은 P50, point는 단일 출력
    plot_pred = te_preds[:, 1] if output_mode == "quantile" else te_preds
    save_pred_plot(y_te, plot_pred, file_tag)
    save_results(fold_results, test_metrics, model_name, phase_tag)

    model_path = os.path.join(RESULTS_DIR, f"{file_tag}_final.keras")
    last_model.save(model_path)
    print(f"  💾 모델 저장: {model_path}")

    return fold_results, test_metrics
