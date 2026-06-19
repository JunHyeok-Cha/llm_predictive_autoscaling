"""
itransformer.py — iTransformer (Keras)
=======================================
핵심: Attention을 시간 축이 아닌 변수(variate) 축에 적용.
  각 변수를 토큰으로 취급 → 변수 간 관계를 Multi-head Attention으로 학습.

구조:
  Input (seq_len, 23)
  → Transpose → (23, seq_len)
  → Dense(d_model) per variate   : 각 변수의 시간 시퀀스를 임베딩
  → InvertedAttentionBlock × 2   : 변수 간 Self-Attention + FFN
  → Flatten → Dense(128, gelu) → Dense(1)

실행:
    python itransformer.py
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
from utils import (
    run_walk_forward, result_exists,
    quantile_loss, sal_loss, SAL_SCENARIOS, QUANTILES,
    N_FEATURES, RESULTS_DIR,
)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
SEQ_LEN    = 336    # lookback 336분 — 긴 lookback이 iTransformer의 강점
D_MODEL    = 128    # 변수별 임베딩 차원
N_HEADS    = 4
N_LAYERS   = 2
D_FF       = 256
DROPOUT    = 0.1
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE   = 8


# ── iTransformer 블록 (Keras Layer) ──────────────────────────────────────────
class InvertedAttentionBlock(layers.Layer):
    """
    입력: (batch, n_vars, d_model)   ← 변수가 토큰
    출력: (batch, n_vars, d_model)

    동작:
      1. Multi-head Self-Attention on variate axis (변수 간 관계)
      2. Feed-Forward (각 변수 독립적으로)
      3. 잔차 연결 + LayerNorm
    """
    def __init__(self, d_model, n_heads, d_ff, dropout, **kwargs):
        super().__init__(**kwargs)
        self.attn   = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=dropout,
        )
        self.ff     = keras.Sequential([
            layers.Dense(d_ff, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
            layers.Dropout(dropout),
        ])
        self.norm1  = layers.LayerNormalization()
        self.norm2  = layers.LayerNormalization()
        self.drop   = layers.Dropout(dropout)

    def call(self, x, training=False):
        # x: (batch, n_vars, d_model)
        # Self-Attention: 변수 간 관계 포착
        attn_out = self.attn(x, x, training=training)
        x = self.norm1(x + self.drop(attn_out, training=training))
        # Feed-Forward: 변수별 독립 처리
        ff_out = self.ff(x, training=training)
        x = self.norm2(x + ff_out)
        return x


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────
def build_itransformer(output_mode="point"):
    """
    output_mode="point"    → Dense(1)  단일 포인트 출력 (Phase 2 SAL)
    output_mode="quantile" → Dense(3)  P10/P50/P90 출력 (Phase 1 분위수)

    처리 흐름:
      (batch, seq_len, n_vars)
      → Transpose → (batch, n_vars, seq_len)
      → Dense: 각 변수를 d_model로 임베딩
      → InvertedAttentionBlock × N_LAYERS
      → Flatten → 출력 헤드
    """
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES), name="input")

    # 전치: 시간 축 ↔ 변수 축 역전
    # (batch, seq_len, n_vars) → (batch, n_vars, seq_len)
    x = layers.Permute((2, 1), name="transpose")(inp)

    # 각 변수의 시간 시퀀스를 d_model로 투영
    # TimeDistributed → 변수 차원을 배치처럼 처리
    x = layers.Dense(D_MODEL, name="input_proj")(x)
    # x: (batch, n_vars, d_model)

    # iTransformer 블록 반복
    for i in range(N_LAYERS):
        x = InvertedAttentionBlock(D_MODEL, N_HEADS, D_FF, DROPOUT,
                                   name=f"itf_block_{i+1}")(x)

    x = layers.LayerNormalization()(x)

    # 모든 변수의 표현을 평탄화하여 출력 헤드로 전달
    x   = layers.Flatten()(x)                         # (batch, n_vars * d_model)
    x = layers.Dense(128, activation="gelu")(x)
    x = layers.Dropout(DROPOUT)(x)
    if output_mode == "quantile":
        out = layers.Dense(3, name="quantile_output")(x)
    else:
        out = layers.Dense(1, name="point_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="iTransformer")
    return model


# ── 실행 ──────────────────────────────────────────────────────────────────────
def _summary(tag, fold_results, test_metrics):
    cv_mae   = np.mean([r["MAE"]   for r in fold_results])
    cv_smape = np.mean([r["SMAPE"] for r in fold_results])
    cv_viol  = np.mean([r["violation_rate"] for r in fold_results])
    cv_sal   = np.mean([r["sal_eval_score"] for r in fold_results])
    print(f"\n[iTransformer · {tag}] 완료 │ "
          f"CV MAE={cv_mae:.4f}  SMAPE={cv_smape:.2f}%  "
          f"ViolRate={cv_viol:.3f}  SAL={cv_sal:.4f} │ "
          f"Test MAE={test_metrics['MAE']:.4f}  "
          f"SAL={test_metrics['sal_eval_score']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  iTransformer — 2-Phase (Quantile + SAL) Walk-Forward CV")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}")
    print(f"  N_LAYERS={N_LAYERS}  D_FF={D_FF}  DROPOUT={DROPOUT}")
    print(f"  BATCH={BATCH_SIZE}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")
    print()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default=None,
                        help="단일 phase만 실행 (quantile|sal_conservative|"
                             "sal_balanced|sal_aggressive). 생략 시 4개 전부.")
    args = parser.parse_args()

    build_itransformer("quantile").summary()
    print()

    FORCE = os.environ.get("FORCE_RETRAIN") == "1"

    # 실행 잡 목록: Phase 1(quantile) + Phase 2(SAL ×3)
    jobs = [("Phase 1: Quantile (Pinball) Loss", "quantile", "quantile",
             lambda: quantile_loss(QUANTILES))]
    for scenario_name, params in SAL_SCENARIOS.items():
        jobs.append((f"Phase 2: SAL Loss — {scenario_name} {params}",
                     f"sal_{scenario_name}", "point",
                     (lambda p=params: sal_loss(**p))))

    if args.phase:   # 단일 phase 격리 실행 (run_all.py가 프로세스 분리용으로 사용)
        jobs = [j for j in jobs if j[1] == args.phase]
        if not jobs:
            raise SystemExit(f"알 수 없는 phase: {args.phase}")

    for title, phase_tag, output_mode, make_loss in jobs:
        print("\n" + "#" * 60 + f"\n  {title}\n" + "#" * 60)
        # resume: 이미 완료된 phase는 건너뜀 (FORCE_RETRAIN=1로 강제 재학습)
        if not FORCE and result_exists("itransformer", phase_tag):
            print(f"  ⏭️  {phase_tag} 결과 존재 — 건너뜀 (재학습: FORCE_RETRAIN=1)")
            continue
        fr, tm = run_walk_forward(
            build_fn=build_itransformer, seq_len=SEQ_LEN, model_name="itransformer",
            output_mode=output_mode, loss_fn=make_loss(), phase_tag=phase_tag,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        _summary(phase_tag, fr, tm)
