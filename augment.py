"""
augment.py — 시계열 데이터 증강
=====================================
Walk-Forward CV 초기 fold(훈련 데이터 부족)의 성능 개선을 위한
시퀀스 레벨 증강 모듈.

배경:
  Fold 1(MAE 0.057) / Fold 4(MAE 0.038)처럼 특정 기간의 성능이 낮은 주요 원인은
  초기 훈련 데이터 부족 및 분포 변화(distribution shift)다.
  시퀀스 레벨 증강으로 다양한 트래픽 패턴을 합성해 일반화 성능을 높인다.

증강 기법:
  jitter    — 연속형 피처에 가우시안 노이즈 (σ=0.03)
  mag_warp  — cubic-spline 기반 smooth 크기 왜곡 (트래픽 레벨 변동 시뮬레이션)
  time_warp — cubic-spline 기반 시간축 국소 왜곡 (패턴 속도 변화 시뮬레이션)
  mixup     — 두 시퀀스 간 선형 보간 (소프트 레이블, 배치 전용)

피처 처리 방침:
  is_zero_fill / hour_sin / hour_cos / weekday_sin / weekday_cos /
  minute_sin / minute_cos / is_weekend  (인덱스 7~14)는 물리적 의미 보존을
  위해 jitter·mag_warp 대상에서 제외.
  time_warp은 전 피처에 동일한 미세 왜곡을 적용 (σ=0.10 수준이므로 허용).

공개 API:
  augment_sequences(X, y, ...)             — numpy 배치 증강 (소규모 fold용)
  make_augmented_train_dataset(df, ...)    — tf.data 온라인 증강 (대규모 fold용)

단독 실행 (효과 확인 및 시각화):
  python augment.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

# ── 피처 인덱스 (utils.py → meta.json feature_cols 순서와 동일) ────────────
# idx  0- 6 : req_count, total_token_throughput, gpt4_ratio, conv_ratio,
#             avg_req_tokens, avg_resp_tokens, failure_rate          (연속형)
# idx  7-14 : is_zero_fill, hour_sin/cos, weekday_sin/cos,
#             minute_sin/cos, is_weekend                             (고정)
# idx 15-22 : lag_1/60/1440/10080, roll_mean/std_60/1440            (연속형)
CONTINUOUS_IDX: list[int] = list(range(0, 7)) + list(range(15, 23))   # 14개
FIXED_IDX:      list[int] = list(range(7, 15))                          # 8개
N_CONT = len(CONTINUOUS_IDX)   # 14


# ══════════════════════════════════════════════════════════════════════════════
# 단일 시퀀스용 내부 헬퍼 (shape: seq_len × n_features)
# — tf.data 온라인 증강에서 1개씩 호출
# ══════════════════════════════════════════════════════════════════════════════

def _jitter_seq(
    x: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """x: (seq_len, n_features)"""
    out = x.copy()
    out[:, CONTINUOUS_IDX] += rng.normal(0.0, sigma, size=(x.shape[0], N_CONT))
    return out


def _mag_warp_seq(
    x: np.ndarray,
    sigma: float,
    n_knots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    cubic-spline으로 생성한 smooth random 커브를 연속형 피처에 곱함.
    같은 시퀀스 내 모든 연속 피처에 동일 커브 적용
    → 전체 트래픽 레벨이 시간에 따라 완만하게 증감하는 패턴 시뮬레이션.
    """
    seq_len = x.shape[0]
    knot_x = np.linspace(0, seq_len - 1, n_knots + 2)
    t = np.arange(seq_len, dtype=np.float64)
    scale = CubicSpline(
        knot_x,
        rng.normal(1.0, sigma, size=n_knots + 2),
    )(t).clip(0.5, 1.5)             # (seq_len,) — 극단 왜곡 방지
    out = x.copy()
    out[:, CONTINUOUS_IDX] = x[:, CONTINUOUS_IDX] * scale[:, np.newaxis]
    return out


def _time_warp_seq(
    x: np.ndarray,
    sigma: float,
    n_knots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    시간축을 smooth random 속도 커브로 왜곡 후 선형 보간으로 재샘플링.
    → 트래픽 피크가 약간 당겨지거나 늦춰지는 패턴 시뮬레이션.
    미세 왜곡(σ=0.10)이므로 시간 인코딩 피처도 동일 변환 적용.
    """
    seq_len, n_feat = x.shape
    knot_x = np.linspace(0, seq_len - 1, n_knots + 2)
    t_orig = np.arange(seq_len, dtype=np.float64)

    speed = CubicSpline(
        knot_x,
        rng.normal(1.0, sigma, size=n_knots + 2).clip(0.1, 3.0),
    )(t_orig).clip(1e-3, None)      # 속도 0 이하 방지

    t_new = np.cumsum(speed)
    t_new = (t_new - t_new[0]) / (t_new[-1] - t_new[0]) * (seq_len - 1)

    out = np.empty_like(x)
    for f in range(n_feat):
        out[:, f] = interp1d(
            t_orig, x[:, f], kind="linear", fill_value="extrapolate"
        )(t_new)
    return out.astype(x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 배치용 내부 헬퍼 (shape: N × seq_len × n_features)
# ══════════════════════════════════════════════════════════════════════════════

def _jitter_batch(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = X.copy()
    out[:, :, CONTINUOUS_IDX] += rng.normal(
        0.0, sigma, size=(X.shape[0], X.shape[1], N_CONT)
    )
    return out


def _mag_warp_batch(
    X: np.ndarray, sigma: float, n_knots: int, rng: np.random.Generator
) -> np.ndarray:
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = _mag_warp_seq(X[i], sigma, n_knots, rng)
    return out


def _time_warp_batch(
    X: np.ndarray, sigma: float, n_knots: int, rng: np.random.Generator
) -> np.ndarray:
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = _time_warp_seq(X[i], sigma, n_knots, rng)
    return out


def _mixup_batch(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Beta(alpha, alpha)에서 λ 샘플링 → 랜덤 쌍 시퀀스를 선형 보간.
    X와 y 모두 보간(소프트 레이블)하여 모델이 중간 패턴을 학습하도록 유도.
    """
    N = len(X)
    lam = rng.beta(alpha, alpha, size=N).astype(np.float32)
    idx = rng.permutation(N)
    X_mix = (
        lam[:, np.newaxis, np.newaxis] * X
        + (1 - lam[:, np.newaxis, np.newaxis]) * X[idx]
    )
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix.astype(np.float32), y_mix.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API 1: numpy 배치 증강
# ══════════════════════════════════════════════════════════════════════════════

_BATCH_METHODS = {
    "jitter":    lambda X, y, kw, rng: (
        _jitter_batch(X, kw["jitter_sigma"], rng), y.copy()
    ),
    "mag_warp":  lambda X, y, kw, rng: (
        _mag_warp_batch(X, kw["magwarp_sigma"], kw["magwarp_knots"], rng), y.copy()
    ),
    "time_warp": lambda X, y, kw, rng: (
        _time_warp_batch(X, kw["timewarp_sigma"], kw["timewarp_knots"], rng), y.copy()
    ),
    "mixup":     lambda X, y, kw, rng: (
        _mixup_batch(X, y, kw["mixup_alpha"], rng)
    ),
}


def augment_sequences(
    X: np.ndarray,
    y: np.ndarray,
    ratio: float = 1.0,
    methods: list[str] | None = None,
    jitter_sigma:   float = 0.03,
    magwarp_sigma:  float = 0.10,
    magwarp_knots:  int   = 4,
    timewarp_sigma: float = 0.10,
    timewarp_knots: int   = 4,
    mixup_alpha:    float = 0.20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    numpy 배치 증강 — 원본 시퀀스에 증강 샘플을 concat하여 반환.

    Parameters
    ----------
    X       : (N, seq_len, n_features)  float32
    y       : (N,)                      float32
    ratio   : 원본 대비 증강 비율. 1.0 → 원본 N개만큼 추가 (총 2N)
    methods : 사용할 기법 목록. None이면 ["jitter", "mag_warp"] 기본 적용.
              지원: "jitter", "mag_warp", "time_warp", "mixup"

    Returns
    -------
    X_total : (N + N*ratio, seq_len, n_features)
    y_total : (N + N*ratio,)

    메모리 주의:
      전체 시퀀스 배열을 메모리에 올리므로 대규모 fold(Fold 5: ~272K샘플)에서는
      make_augmented_train_dataset() 사용을 권장.
    """
    if methods is None:
        methods = ["jitter", "mag_warp"]

    unknown = set(methods) - set(_BATCH_METHODS)
    if unknown:
        raise ValueError(f"지원하지 않는 기법: {unknown}. 선택 가능: {list(_BATCH_METHODS)}")

    rng = np.random.default_rng(seed)
    N = len(X)
    n_total_aug = int(N * ratio)
    n_per_method = max(1, n_total_aug // len(methods))

    kw = dict(
        jitter_sigma=jitter_sigma,
        magwarp_sigma=magwarp_sigma,
        magwarp_knots=magwarp_knots,
        timewarp_sigma=timewarp_sigma,
        timewarp_knots=timewarp_knots,
        mixup_alpha=mixup_alpha,
    )

    X_parts: list[np.ndarray] = [X]
    y_parts: list[np.ndarray] = [y]

    for method in methods:
        idx = rng.integers(0, N, size=n_per_method)
        Xa, ya = _BATCH_METHODS[method](X[idx].copy(), y[idx].copy(), kw, rng)
        X_parts.append(Xa.astype(np.float32))
        y_parts.append(ya.astype(np.float32))

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API 2: tf.data 온라인 증강 (메모리 효율적)
# ══════════════════════════════════════════════════════════════════════════════

def make_augmented_train_dataset(
    df,
    seq_len: int,
    batch_size: int,
    aug_prob: float = 0.5,
    methods: list[str] | None = None,
    jitter_sigma:   float = 0.03,
    magwarp_sigma:  float = 0.10,
    magwarp_knots:  int   = 4,
    timewarp_sigma: float = 0.10,
    timewarp_knots: int   = 4,
    seed: int = 42,
):
    """
    tf.data 제너레이터 기반 온라인 증강.
    원시 CSV 배열(~12 MB)만 메모리에 올리고 배치 단위로 증강 시퀀스를 생성.
    → Fold 3~5처럼 훈련 데이터가 큰 경우에도 메모리 안전.

    mixup은 랜덤 쌍이 필요해 온라인 방식과 맞지 않으므로 제외.
    사용 가능 methods: "jitter", "mag_warp", "time_warp"

    Parameters
    ----------
    df         : 훈련용 pandas DataFrame (utils.load_fold(fold)[0])
    aug_prob   : 각 샘플에 증강을 적용할 확률 (0.0이면 원본과 동일)
    methods    : None이면 ["jitter", "mag_warp"]

    Returns
    -------
    (tf.data.Dataset, n_samples: int)
      Dataset: (seq_len, n_features), () 형태의 배치 스트림
    """
    import tensorflow as tf
    from utils import FEATURE_COLS, N_FEATURES

    if methods is None:
        methods = ["jitter", "mag_warp"]
    online_ok = {"jitter", "mag_warp", "time_warp"}
    bad = set(methods) - online_ok
    if bad:
        raise ValueError(
            f"온라인 증강에서 {bad}는 지원 불가. 사용 가능: {online_ok}"
        )

    X_arr = df[FEATURE_COLS].values.astype(np.float32)
    y_arr = df["target"].values.astype(np.float32)

    valid = np.where(
        ~(np.isnan(y_arr[seq_len:]) | np.isnan(X_arr[seq_len:]).any(axis=1))
    )[0] + seq_len
    n_samples = len(valid)

    kw = dict(
        jitter_sigma=jitter_sigma,
        magwarp_sigma=magwarp_sigma,
        magwarp_knots=magwarp_knots,
        timewarp_sigma=timewarp_sigma,
        timewarp_knots=timewarp_knots,
    )

    _online_fns = {
        "jitter":    lambda x, rng: _jitter_seq(x, kw["jitter_sigma"], rng),
        "mag_warp":  lambda x, rng: _mag_warp_seq(x, kw["magwarp_sigma"], kw["magwarp_knots"], rng),
        "time_warp": lambda x, rng: _time_warp_seq(x, kw["timewarp_sigma"], kw["timewarp_knots"], rng),
    }

    rng = np.random.default_rng(seed)

    def _gen():
        idx = valid.copy()
        np.random.shuffle(idx)
        for i in idx:
            x = X_arr[i - seq_len:i]   # (seq_len, n_features)
            y = y_arr[i]
            if rng.random() < aug_prob:
                method = methods[rng.integers(len(methods))]
                x = _online_fns[method](x, rng)
            yield x, y

    ds = tf.data.Dataset.from_generator(
        _gen,
        output_signature=(
            tf.TensorSpec(shape=(seq_len, N_FEATURES), dtype=tf.float32),
            tf.TensorSpec(shape=(),                    dtype=tf.float32),
        ),
    )
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE), n_samples


# ══════════════════════════════════════════════════════════════════════════════
# 단독 실행: 효과 분석 및 시각화
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils import make_sequences, load_fold, RESULTS_DIR, FEATURE_COLS

    print("=" * 60)
    print("  데이터 증강 효과 분석 (Fold 1 — 가장 적은 훈련 데이터)")
    print("=" * 60)

    SEQ_LEN = 168
    tr_df, _ = load_fold(1)
    X, y = make_sequences(tr_df, SEQ_LEN)
    print(f"\n원본  X={X.shape}  y: mean={y.mean():.4f}  std={y.std():.4f}\n")

    ALL_METHODS = ["jitter", "mag_warp", "time_warp", "mixup"]

    # ── 기법별 샘플 수 및 분포 변화 ──────────────────────────────────────────
    print(f"{'기법':12s}  {'증강 후 샘플':>12s}  {'y mean':>10s}  {'y std':>10s}  {'X[0] mean':>12s}")
    print("-" * 65)
    for method in ALL_METHODS:
        Xa, ya = augment_sequences(X, y, ratio=1.0, methods=[method], seed=0)
        print(
            f"  {method:10s}  {len(Xa):>12,}  "
            f"{ya.mean():>10.4f}  {ya.std():>10.4f}  "
            f"{Xa[:, :, 0].mean():>12.4f}"
        )

    # ── 기본 설정 (jitter + mag_warp) 총 효과 ───────────────────────────────
    X_aug, y_aug = augment_sequences(X, y, ratio=1.0)
    print(f"\n기본 증강 (jitter + mag_warp):")
    print(f"  {X.shape[0]:,} → {X_aug.shape[0]:,} 샘플 (+{X_aug.shape[0]-X.shape[0]:,}개 추가)")

    # ── 시각화: 기법별 req_count (idx 0) 비교 ───────────────────────────────
    SAMPLE = 200   # 시각화할 시퀀스 인덱스
    t = np.arange(SEQ_LEN)
    feat_idx = 0   # req_count

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, method in zip(axes.flat, ALL_METHODS):
        Xa, _ = augment_sequences(
            X[[SAMPLE]], y[[SAMPLE]], ratio=1.0, methods=[method], seed=42
        )
        ax.plot(t, X[SAMPLE, :, feat_idx],  label="원본",  lw=1.8,
                color="steelblue")
        ax.plot(t, Xa[1, :, feat_idx],      label=method,  lw=1.2,
                color="tomato", alpha=0.85, linestyle="--")
        ax.set_title(f"{method}  —  {FEATURE_COLS[feat_idx]}")
        ax.set_xlabel("Time step (min)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"BiLSTM 데이터 증강: 원본 vs 각 기법 (sample #{SAMPLE}, feature: req_count)",
        fontsize=12,
    )
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "augmentation_preview.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\n  시각화 저장: {out_path}")

    # ── 통합 가이드 출력 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  통합 방법")
    print("=" * 60)
    print("""
[소규모 fold — numpy 배치 방식]
  from augment import augment_sequences
  X_tr, y_tr = make_sequences(tr_df, seq_len)
  X_tr, y_tr = augment_sequences(X_tr, y_tr, ratio=1.0,
                                 methods=["jitter", "mag_warp"])
  tr_ds = (tf.data.Dataset
           .from_tensor_slices((X_tr, y_tr))
           .shuffle(len(X_tr))
           .batch(batch_size)
           .prefetch(tf.data.AUTOTUNE))

[대규모 fold — tf.data 온라인 방식 (메모리 효율)]
  from augment import make_augmented_train_dataset
  tr_ds, n_tr = make_augmented_train_dataset(
      tr_df, seq_len, batch_size, aug_prob=0.5,
      methods=["jitter", "mag_warp"])
""")
