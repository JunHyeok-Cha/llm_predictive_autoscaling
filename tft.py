"""
tft.py — Temporal Fusion Transformer (Keras)
=============================================
핵심 특징:
  1. 과거 관측값 / 미래 알려진 값(시간 인코딩) 분리 처리
  2. Variable Selection Network (VSN): 피처별 중요도 가중치 학습
  3. Gated Residual Network (GRN): 비선형 변환 + 잔차 연결
  4. LSTM 인코더-디코더
  5. Multi-head Attention: 장기 의존성 포착
  6. 분위수 출력: P10 / P50 / P90 동시 예측

실행:
    python tft.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from utils import (
    PREP_DIR, RESULTS_DIR, N_FOLDS,
    PAST_ONLY_COLS, KNOWN_FUTURE_COLS,
    N_PAST, N_KNOWN, PRED_HORIZON,
    load_fold, load_test,
    make_tft_sequences, compute_metrics,
    save_loss_curve, save_results,
    TqdmCallback,
)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
SEQ_LEN    = 168        # lookback 168분
HIDDEN     = 64         # GRN / LSTM hidden 차원
N_HEADS    = 4
DROPOUT    = 0.1
QUANTILES  = [0.1, 0.5, 0.9]   # P10, P50, P90
BATCH_SIZE = 128
MAX_EPOCHS = 60
PATIENCE   = 8


# ── GRN (Gated Residual Network) ─────────────────────────────────────────────
class GRN(layers.Layer):
    """
    TFT 기본 비선형 블록.
    출력 = LayerNorm(x_proj + GLU(ELU(W1·x) · W2·x))
    """
    def __init__(self, d_hidden, d_out, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.fc1    = layers.Dense(d_hidden, activation="elu")
        self.fc2    = layers.Dense(d_out * 2)          # GLU 위해 2배
        self.fc_res = layers.Dense(d_out)              # residual 투영
        self.norm   = layers.LayerNormalization()
        self.drop   = layers.Dropout(dropout)

    def call(self, x, training=False):
        residual = self.fc_res(x)
        h = self.fc1(x)
        h = self.drop(self.fc2(h), training=training)
        # GLU: 절반은 값, 절반은 게이트
        v, g = tf.split(h, 2, axis=-1)
        h = v * tf.sigmoid(g)
        return self.norm(residual + h)


# ── Variable Selection Network ────────────────────────────────────────────────
class VSN(layers.Layer):
    """
    각 피처에 softmax 중요도 가중치를 부여.
    출력: 가중합된 표현 (d_model 차원)
    """
    def __init__(self, n_vars, d_model, hidden, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_vars   = n_vars
        # 변수별 GRN (입력 차원 1 → d_model)
        self.var_grns = [GRN(hidden, d_model, dropout) for _ in range(n_vars)]
        # 전체 → softmax 가중치
        self.weight_grn = GRN(hidden, n_vars, dropout)

    def call(self, x, training=False):
        # x: (..., n_vars)
        # 변수별 표현 생성
        var_outs = tf.stack(
            [grn(x[..., i:i+1], training=training)
             for i, grn in enumerate(self.var_grns)],
            axis=-2
        )  # (..., n_vars, d_model)

        # softmax 가중치
        weights = tf.nn.softmax(
            self.weight_grn(x, training=training), axis=-1
        )  # (..., n_vars)

        # 가중합
        out = tf.reduce_sum(
            var_outs * weights[..., tf.newaxis], axis=-2
        )  # (..., d_model)
        return out, weights


# ── TFT 모델 빌더 ─────────────────────────────────────────────────────────────
def build_tft():
    """
    입력:
      past_input:   (batch, seq_len, N_PAST)
      future_input: (batch, PRED_HORIZON, N_KNOWN)

    출력:
      (batch, 3)  ← [P10, P50, P90]
    """
    d_model = HIDDEN

    # ── 입력 ──
    past_inp   = keras.Input(shape=(SEQ_LEN,    N_PAST),  name="past_input")
    future_inp = keras.Input(shape=(PRED_HORIZON, N_KNOWN), name="future_input")

    # ── Variable Selection ──
    # 시간 차원을 배치처럼 처리하기 위해 reshape
    B_past   = tf.shape(past_inp)[0]
    B_future = tf.shape(future_inp)[0]

    past_flat   = tf.reshape(past_inp,   (-1, N_PAST))
    future_flat = tf.reshape(future_inp, (-1, N_KNOWN))

    vsn_past   = VSN(N_PAST,  d_model, HIDDEN * 2, DROPOUT, name="vsn_past")
    vsn_future = VSN(N_KNOWN, d_model, HIDDEN * 2, DROPOUT, name="vsn_future")

    enc_flat, _ = vsn_past(past_flat)
    dec_flat, _ = vsn_future(future_flat)

    # reshape back: (batch, seq_len, d_model) / (batch, horizon, d_model)
    enc_in = layers.Reshape((SEQ_LEN,     d_model))(enc_flat)
    dec_in = layers.Reshape((PRED_HORIZON, d_model))(dec_flat)

    # ── LSTM 인코더 ──
    enc_out, state_h, state_c = layers.LSTM(
        HIDDEN, return_sequences=True, return_state=True, name="enc_lstm"
    )(enc_in)

    # ── LSTM 디코더 (인코더 마지막 상태 전달) ──
    dec_out = layers.LSTM(
        HIDDEN, return_sequences=True, name="dec_lstm"
    )(dec_in, initial_state=[state_h, state_c])

    # ── Multi-head Attention (디코더가 인코더 전체를 참조) ──
    attn_out = layers.MultiHeadAttention(
        num_heads=N_HEADS, key_dim=HIDDEN // N_HEADS,
        dropout=DROPOUT, name="cross_attn"
    )(query=dec_out, value=enc_out, key=enc_out)
    attn_out = layers.LayerNormalization()(dec_out + attn_out)

    # ── 마지막 디코더 타임스텝 → 출력 ──
    final = attn_out[:, -1, :]           # (batch, hidden)

    # GRN 후처리
    grn_out = GRN(HIDDEN * 2, HIDDEN, DROPOUT, name="post_grn")(final)
    grn_out = layers.Dropout(DROPOUT)(grn_out)

    # 분위수 출력: 한 번에 3개 (P10, P50, P90)
    out = layers.Dense(len(QUANTILES), name="quantile_output")(grn_out)

    model = keras.Model(
        inputs=[past_inp, future_inp],
        outputs=out,
        name="TFT"
    )
    return model


# ── 분위수 손실 (Pinball Loss) ────────────────────────────────────────────────
def quantile_loss(quantiles):
    """
    각 분위수 τ에 대한 pinball loss 합산.
    τ=0.9: 과소예측 패널티 큼 → P90이 실제보다 낮을 때 비용 커짐.
    """
    q = tf.constant(quantiles, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: (batch,)  y_pred: (batch, n_q)
        y_true = tf.expand_dims(y_true, -1)          # (batch, 1)
        err    = y_true - y_pred                      # (batch, n_q)
        loss   = tf.maximum(q * err, (q - 1.0) * err)
        return tf.reduce_mean(loss)

    return loss_fn


# ── TFT 전용 Walk-Forward CV ──────────────────────────────────────────────────
def run_tft_walk_forward():
    from tqdm import tqdm

    fold_results = []
    last_model   = None

    fold_bar = tqdm(
        range(1, N_FOLDS + 1),
        desc="[TFT] 전체 진행",
        unit="fold",
        position=0,
        colour="green",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} fold [{elapsed}] {postfix}",
    )

    for fold in fold_bar:
        fold_bar.set_postfix(현재=f"Fold {fold}")
        print(f"\n{'─'*55}")
        print(f"  [TFT]  Fold {fold} / {N_FOLDS}")
        print(f"{'─'*55}")

        tr_df, vl_df = load_fold(fold)
        past_tr, fut_tr, y_tr = make_tft_sequences(tr_df, SEQ_LEN)
        past_vl, fut_vl, y_vl = make_tft_sequences(vl_df, SEQ_LEN)
        print(f"  데이터  train: past={past_tr.shape} fut={fut_tr.shape}  "
              f"val: past={past_vl.shape}")

        model = build_tft()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=quantile_loss(QUANTILES),
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE,
                restore_best_weights=True, verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0,
            ),
            TqdmCallback(
                total_epochs=MAX_EPOCHS,
                fold=fold,
                model_name="tft",
            ),
        ]

        history = model.fit(
            [past_tr, fut_tr], y_tr,
            validation_data=([past_vl, fut_vl], y_vl),
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

        vl_preds = model.predict([past_vl, fut_vl], verbose=0)  # (N, 3)
        metrics  = compute_metrics(y_vl, vl_preds[:, 1])        # P50 기준
        metrics["fold"] = fold
        fold_results.append(metrics)
        print(f"\n  ✅ Fold {fold} 결과 (P50) │ "
              f"MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  "
              f"MAPE={metrics['MAPE']:.2f}%")

        save_loss_curve(history, fold, "tft")
        last_model = model

    fold_bar.close()

    # ── 최종 테스트 ──────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  [TFT]  Final Test")
    print(f"{'═'*55}")
    test_df = load_test()
    past_te, fut_te, y_te = make_tft_sequences(test_df, SEQ_LEN)
    te_preds = last_model.predict([past_te, fut_te], verbose=0)  # (N, 3)

    test_metrics = compute_metrics(y_te, te_preds[:, 1])   # P50 기준
    p90_mae = float(np.mean(np.abs(y_te - te_preds[:, 2])))
    print(f"  🏁 Test (P50) │ MAE={test_metrics['MAE']:.4f}  "
          f"RMSE={test_metrics['RMSE']:.4f}  MAPE={test_metrics['MAPE']:.2f}%")
    print(f"     Test (P90) │ MAE={p90_mae:.4f}  ← Pod 용량 결정 기준")

    # P10/P50/P90 밴드 시각화
    _save_tft_pred_plot(y_te, te_preds)
    save_results(fold_results, test_metrics, "tft")

    # 모델 + 분위수 예측값 저장
    model_path = os.path.join(RESULTS_DIR, "tft_final.keras")
    last_model.save(model_path)
    np.save(os.path.join(RESULTS_DIR, "tft_test_quantiles.npy"), te_preds)
    np.save(os.path.join(RESULTS_DIR, "tft_test_trues.npy"),     y_te)
    print(f"  모델 저장: {model_path}")
    print(f"  분위수 저장: results/tft_test_quantiles.npy  {te_preds.shape}")

    return fold_results, test_metrics, te_preds, y_te


def _save_tft_pred_plot(y_true, preds_all, n=2000):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(min(n, len(y_true))),
                    preds_all[:n, 0], preds_all[:n, 2],
                    alpha=0.25, color="steelblue", label="P10–P90 band")
    ax.plot(y_true[:n],         color="black",     lw=0.8, label="Actual")
    ax.plot(preds_all[:n, 1],   color="steelblue", lw=0.8, label="P50 (median)")
    ax.set_title("TFT — Final Test Predictions with Uncertainty Band")
    ax.set_xlabel("Time step (min)"); ax.set_ylabel("req_count (normalized)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "tft_test_pred.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  예측 그래프 저장: {path}")


# ── 실행 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  TFT (Temporal Fusion Transformer) — Walk-Forward CV (Keras)")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}  HIDDEN={HIDDEN}  N_HEADS={N_HEADS}")
    print(f"  QUANTILES={QUANTILES}  DROPOUT={DROPOUT}")
    print(f"  N_PAST={N_PAST}  N_KNOWN={N_KNOWN}  BATCH={BATCH_SIZE}")
    print()

    sample = build_tft()
    sample.summary()
    print()

    fold_results, test_metrics, te_preds, y_te = run_tft_walk_forward()

    cv_mae  = np.mean([r["MAE"]  for r in fold_results])
    cv_rmse = np.mean([r["RMSE"] for r in fold_results])
    cv_mape = np.mean([r["MAPE"] for r in fold_results])

    print("\n[TFT] 학습 완료")
    print(f"  CV 평균 MAE  : {cv_mae:.4f}")
    print(f"  CV 평균 RMSE : {cv_rmse:.4f}")
    print(f"  CV 평균 MAPE : {cv_mape:.2f}%")
    print(f"  Test MAE (P50) : {test_metrics['MAE']:.4f}")
    print(f"  Test RMSE(P50) : {test_metrics['RMSE']:.4f}")
    print(f"  Test MAPE(P50) : {test_metrics['MAPE']:.2f}%")

    interval = te_preds[:, 2] - te_preds[:, 0]
    print(f"\n  P90-P10 평균 간격: {interval.mean():.4f} (normalized scale)")
    print(f"  → 이 값이 안전 마진(ε) 설계의 통계적 근거")
