"""
bilstm.py — Bidirectional LSTM (Keras)
=======================================
구조:
  Input (seq_len, 23)
  → Bidirectional LSTM (128) + Dropout(0.2)
  → Bidirectional LSTM (64)  + Dropout(0.2)
  → Dense(64, relu)
  → Dense(1)

실행:
    python bilstm.py
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import (
    run_walk_forward,
    N_FEATURES, RESULTS_DIR,
)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
SEQ_LEN    = 168   # lookback 168분
HIDDEN1    = 128
HIDDEN2    = 64
DROPOUT    = 0.2
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE   = 8


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────
def build_bilstm():
    """
    호출할 때마다 새 모델 인스턴스를 반환 (fold마다 초기화).

    구조 설명:
      Bidirectional LSTM — 정방향 + 역방향으로 시퀀스를 동시에 읽어
      각 시점의 과거·미래 문맥을 모두 활용한 hidden state를 만든다.
      2레이어 쌓기: 첫 번째 층이 저수준 패턴(단기 변동)을 포착하고,
      두 번째 층이 고수준 패턴(주기성, 트렌드)을 포착한다.
    """
    inp = keras.Input(shape=(SEQ_LEN, N_FEATURES), name="input")

    # Layer 1: BiLSTM — return_sequences=True 로 다음 LSTM 에 전달
    x = layers.Bidirectional(
        layers.LSTM(HIDDEN1, return_sequences=True),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(DROPOUT)(x)

    # Layer 2: BiLSTM — return_sequences=False 로 마지막 타임스텝만
    x = layers.Bidirectional(
        layers.LSTM(HIDDEN2, return_sequences=False),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(DROPOUT)(x)

    # 출력 헤드
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BiLSTM")
    return model


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Bidirectional LSTM — Walk-Forward 5-Fold CV (Keras)")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}  HIDDEN={HIDDEN1}/{HIDDEN2}")
    print(f"  DROPOUT={DROPOUT}  BATCH={BATCH_SIZE}  MAX_EPOCHS={MAX_EPOCHS}")
    print()

    # 모델 구조 확인
    sample = build_bilstm()
    sample.summary()
    print()

    fold_results, test_metrics = run_walk_forward(
        build_fn   = build_bilstm,
        seq_len    = SEQ_LEN,
        model_name = "bilstm",
        batch_size = BATCH_SIZE,
        max_epochs = MAX_EPOCHS,
        patience   = PATIENCE,
    )

    cv_mae  = np.mean([r["MAE"]  for r in fold_results])
    cv_rmse = np.mean([r["RMSE"] for r in fold_results])
    cv_mape = np.mean([r["MAPE"] for r in fold_results])

    print("\n[BiLSTM] 학습 완료")
    print(f"  CV 평균 MAE  : {cv_mae:.4f}")
    print(f"  CV 평균 RMSE : {cv_rmse:.4f}")
    print(f"  CV 평균 MAPE : {cv_mape:.2f}%")
    print(f"  Test MAE     : {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE    : {test_metrics['RMSE']:.4f}")
    print(f"  Test MAPE    : {test_metrics['MAPE']:.2f}%")
