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

import os
from utils import (
    run_walk_forward, result_exists,
    quantile_loss, sal_loss, SAL_SCENARIOS, QUANTILES,
    N_FEATURES, RESULTS_DIR,
)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
SEQ_LEN    = 168   # lookback 168분
HIDDEN1    = 64
HIDDEN2    = 32
DROPOUT    = 0.4
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE   = 12


# ── 모델 빌더 ─────────────────────────────────────────────────────────────────
def build_bilstm(output_mode="point"):
    """
    호출할 때마다 새 모델 인스턴스를 반환 (fold마다 초기화).
    output_mode="point"    → Dense(1)  단일 포인트 출력 (Phase 2 SAL)
    output_mode="quantile" → Dense(3)  P10/P50/P90 출력 (Phase 1 분위수)

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
    x = layers.Dense(64, activation="relu")(x)
    if output_mode == "quantile":
        out = layers.Dense(3, name="quantile_output")(x)
    else:
        out = layers.Dense(1, name="point_output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="BiLSTM")
    return model


# ── 실행 ──────────────────────────────────────────────────────────────────────
def _summary(tag, fold_results, test_metrics):
    cv_mae   = np.mean([r["MAE"]   for r in fold_results])
    cv_smape = np.mean([r["SMAPE"] for r in fold_results])
    cv_viol  = np.mean([r["violation_rate"] for r in fold_results])
    cv_sal   = np.mean([r["sal_eval_score"] for r in fold_results])
    print(f"\n[BiLSTM · {tag}] 완료 │ "
          f"CV MAE={cv_mae:.4f}  SMAPE={cv_smape:.2f}%  "
          f"ViolRate={cv_viol:.3f}  SAL={cv_sal:.4f} │ "
          f"Test MAE={test_metrics['MAE']:.4f}  "
          f"SAL={test_metrics['sal_eval_score']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Bidirectional LSTM — 2-Phase (Quantile + SAL) Walk-Forward CV")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}  HIDDEN={HIDDEN1}/{HIDDEN2}  DROPOUT={DROPOUT}")
    print(f"  BATCH={BATCH_SIZE}  MAX_EPOCHS={MAX_EPOCHS}  PATIENCE={PATIENCE}")
    print()

    # 모델 구조 확인 (quantile 헤드)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default=None,
                        help="단일 phase만 실행 (quantile|sal_conservative|"
                             "sal_balanced|sal_aggressive). 생략 시 4개 전부.")
    args = parser.parse_args()

    build_bilstm("quantile").summary()
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
        if not FORCE and result_exists("bilstm", phase_tag):
            print(f"  ⏭️  {phase_tag} 결과 존재 — 건너뜀 (재학습: FORCE_RETRAIN=1)")
            continue
        fr, tm = run_walk_forward(
            build_fn=build_bilstm, seq_len=SEQ_LEN, model_name="bilstm",
            output_mode=output_mode, loss_fn=make_loss(), phase_tag=phase_tag,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
        )
        _summary(phase_tag, fr, tm)
