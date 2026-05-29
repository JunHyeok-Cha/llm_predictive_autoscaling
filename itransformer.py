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

from utils import run_walk_forward, N_FEATURES, RESULTS_DIR

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
def build_itransformer():
    """
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
    x   = layers.Dense(128, activation="gelu")(x)
    x   = layers.Dropout(DROPOUT)(x)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="iTransformer")
    return model


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  iTransformer — Walk-Forward 5-Fold CV (Keras)")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}  D_MODEL={D_MODEL}  N_HEADS={N_HEADS}")
    print(f"  N_LAYERS={N_LAYERS}  D_FF={D_FF}  DROPOUT={DROPOUT}")
    print(f"  BATCH={BATCH_SIZE}  MAX_EPOCHS={MAX_EPOCHS}")
    print()

    sample = build_itransformer()
    sample.summary()
    print()

    fold_results, test_metrics = run_walk_forward(
        build_fn   = build_itransformer,
        seq_len    = SEQ_LEN,
        model_name = "itransformer",
        batch_size = BATCH_SIZE,
        max_epochs = MAX_EPOCHS,
        patience   = PATIENCE,
    )

    cv_mae  = np.mean([r["MAE"]  for r in fold_results])
    cv_rmse = np.mean([r["RMSE"] for r in fold_results])
    cv_mape = np.mean([r["MAPE"] for r in fold_results])

    print("\n[iTransformer] 학습 완료")
    print(f"  CV 평균 MAE  : {cv_mae:.4f}")
    print(f"  CV 평균 RMSE : {cv_rmse:.4f}")
    print(f"  CV 평균 MAPE : {cv_mape:.2f}%")
    print(f"  Test MAE     : {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE    : {test_metrics['RMSE']:.4f}")
    print(f"  Test MAPE    : {test_metrics['MAPE']:.2f}%")
