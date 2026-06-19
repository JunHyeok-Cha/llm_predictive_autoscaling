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
    make_tft_sequences, make_tft_train_dataset,
    compute_metrics, evaluate_predictions,
    save_loss_curve, save_results, result_exists,
    quantile_loss, sal_loss, SAL_SCENARIOS, QUANTILES,
    TqdmCallback,
)

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
SEQ_LEN    = 168        # lookback 168분
HIDDEN     = 64         # GRN / LSTM hidden 차원
N_HEADS    = 4
DROPOUT    = 0.1
# QUANTILES(=[0.1,0.5,0.9])는 utils에서 import — BiLSTM/iTransformer와 공통
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
def build_tft(output_mode="quantile"):
    """
    입력:
      past_input:   (batch, seq_len, N_PAST)
      future_input: (batch, PRED_HORIZON, N_KNOWN)

    출력:
      output_mode="quantile" → (batch, 3)  ← [P10, P50, P90]  (Phase 1)
      output_mode="point"    → (batch, 1)  ← 단일 포인트       (Phase 2 SAL)
    """
    d_model = HIDDEN

    # ── 입력 ──
    past_inp   = keras.Input(shape=(SEQ_LEN,    N_PAST),  name="past_input")
    future_inp = keras.Input(shape=(PRED_HORIZON, N_KNOWN), name="future_input")

    # ── Variable Selection ──
    # VSN.call은 ... (ellipsis) 인덱싱을 써서 임의의 선행 차원을 지원하므로,
    # 3D 입력 (batch, time, n_vars)에 그대로 적용 가능 → flatten/reshape 불필요.
    # (Keras 3에서는 Input KerasTensor에 raw tf.shape/tf.reshape를 못 씀)
    vsn_past   = VSN(N_PAST,  d_model, HIDDEN * 2, DROPOUT, name="vsn_past")
    vsn_future = VSN(N_KNOWN, d_model, HIDDEN * 2, DROPOUT, name="vsn_future")

    enc_in, _ = vsn_past(past_inp)      # (batch, seq_len,      d_model)
    dec_in, _ = vsn_future(future_inp)  # (batch, pred_horizon, d_model)

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

    # 출력 헤드: quantile=3개(P10/P50/P90), point=단일
    if output_mode == "quantile":
        out = layers.Dense(len(QUANTILES), name="quantile_output")(grn_out)
    else:
        out = layers.Dense(1, name="point_output")(grn_out)

    model = keras.Model(
        inputs=[past_inp, future_inp],
        outputs=out,
        name="TFT"
    )
    return model


# ── TFT 전용 Walk-Forward CV (past/future 분리 입력) ─────────────────────────
def run_tft_walk_forward(
    output_mode: str,      # "point" | "quantile"
    loss_fn,
    phase_tag: str,         # "quantile" | "sal_conservative" | "sal_balanced" | "sal_aggressive"
    eval_sal_params=None,   # 기본값 SAL_SCENARIOS["balanced"] (공통 평가 기준)
):
    """
    입력 구조(past/future 분리)가 달라 별도 함수를 유지하되,
    output_mode/loss_fn/phase_tag/평가지표/파일명 규칙은 run_walk_forward와 일치.
    """
    import gc
    from tqdm import tqdm

    # Phase 경계 메모리 정리 (이전 phase의 그래프/세션 해제 → OOM 완화)
    keras.backend.clear_session()
    gc.collect()

    if eval_sal_params is None:
        eval_sal_params = SAL_SCENARIOS["balanced"]

    file_tag = f"tft_{phase_tag}"
    disp     = f"tft-{phase_tag}"

    fold_results = []
    last_model   = None

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

        if last_model is not None:
            del last_model
            last_model = None
            keras.backend.clear_session()

        tr_df, vl_df = load_fold(fold)
        # 훈련: 제너레이터 기반 — 전체 dense 배열을 메모리에 안 올림 (OOM 회피)
        tr_ds, n_tr = make_tft_train_dataset(tr_df, SEQ_LEN, BATCH_SIZE)
        # 검증: numpy — val 세트는 작아서 허용
        past_vl, fut_vl, y_vl = make_tft_sequences(vl_df, SEQ_LEN)
        print(f"  데이터  train_samples={n_tr}  val: past={past_vl.shape}")

        del tr_df, vl_df   # raw CSV 즉시 해제
        gc.collect()

        model = build_tft(output_mode)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss=loss_fn,
            jit_compile=False,   # XLA 끔: Turing(GTX 16xx) cuBLAS gemm autotuner 실패 회피
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
                model_name=disp,
            ),
        ]

        history = model.fit(
            tr_ds,
            validation_data=([past_vl, fut_vl], y_vl),
            epochs=MAX_EPOCHS,
            callbacks=callbacks,
            verbose=0,
        )

        vl_preds = model.predict([past_vl, fut_vl], verbose=0)
        if output_mode == "point":
            vl_preds = vl_preds.reshape(-1)
        metrics  = evaluate_predictions(y_vl, vl_preds, output_mode, eval_sal_params)
        metrics["fold"] = fold
        fold_results.append(metrics)
        print(f"\n  ✅ Fold {fold} 결과 │ "
              f"MAE={metrics['MAE']:.4f}  RMSE={metrics['RMSE']:.4f}  "
              f"SMAPE={metrics['SMAPE']:.2f}%  "
              f"ViolRate={metrics['violation_rate']:.3f}  "
              f"SAL={metrics['sal_eval_score']:.4f}")

        save_loss_curve(history, fold, file_tag)
        last_model = model

        del tr_ds, past_vl, fut_vl, y_vl   # fold 메모리 해제
        gc.collect()

    fold_bar.close()

    # ── 최종 테스트 ──────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  [{disp.upper()}]  Final Test")
    print(f"{'═'*55}")
    test_df = load_test()
    past_te, fut_te, y_te = make_tft_sequences(test_df, SEQ_LEN)
    te_preds = last_model.predict([past_te, fut_te], verbose=0)
    if output_mode == "point":
        te_preds = te_preds.reshape(-1)

    test_metrics = evaluate_predictions(y_te, te_preds, output_mode, eval_sal_params)
    print(f"  🏁 Test 결과 │ "
          f"MAE={test_metrics['MAE']:.4f}  RMSE={test_metrics['RMSE']:.4f}  "
          f"SMAPE={test_metrics['SMAPE']:.2f}%  "
          f"ViolRate={test_metrics['violation_rate']:.3f}  "
          f"SAL={test_metrics['sal_eval_score']:.4f}")

    # 예측 시각화: quantile은 P10–P90 밴드, point는 단일 라인
    if output_mode == "quantile":
        _save_tft_band_plot(y_te, te_preds, file_tag)
    else:
        _save_tft_point_plot(y_te, te_preds, file_tag)
    save_results(fold_results, test_metrics, "tft", phase_tag)

    # 모델 저장 (+ quantile일 때 분위수 예측 배열 저장)
    model_path = os.path.join(RESULTS_DIR, f"{file_tag}_final.keras")
    last_model.save(model_path)
    print(f"  💾 모델 저장: {model_path}")
    if output_mode == "quantile":
        np.save(os.path.join(RESULTS_DIR, f"{file_tag}_test_quantiles.npy"), te_preds)
        np.save(os.path.join(RESULTS_DIR, f"{file_tag}_test_trues.npy"),     y_te)
        print(f"  분위수 저장: results/{file_tag}_test_quantiles.npy  {te_preds.shape}")

    return fold_results, test_metrics


def _save_tft_band_plot(y_true, preds_all, file_tag, n=2000):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(range(min(n, len(y_true))),
                    preds_all[:n, 0], preds_all[:n, 2],
                    alpha=0.25, color="steelblue", label="P10–P90 band")
    ax.plot(y_true[:n],         color="black",     lw=0.8, label="Actual")
    ax.plot(preds_all[:n, 1],   color="steelblue", lw=0.8, label="P50 (median)")
    ax.set_title(f"{file_tag} — Final Test Predictions with Uncertainty Band")
    ax.set_xlabel("Time step (min)"); ax.set_ylabel("req_count (normalized)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{file_tag}_test_pred.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  예측 그래프 저장: {path}")


def _save_tft_point_plot(y_true, y_pred, file_tag, n=2000):
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


# ── 실행 ──────────────────────────────────────────────────────────────────────
def _summary(tag, fold_results, test_metrics):
    cv_mae   = np.mean([r["MAE"]   for r in fold_results])
    cv_smape = np.mean([r["SMAPE"] for r in fold_results])
    cv_viol  = np.mean([r["violation_rate"] for r in fold_results])
    cv_sal   = np.mean([r["sal_eval_score"] for r in fold_results])
    print(f"\n[TFT · {tag}] 완료 │ "
          f"CV MAE={cv_mae:.4f}  SMAPE={cv_smape:.2f}%  "
          f"ViolRate={cv_viol:.3f}  SAL={cv_sal:.4f} │ "
          f"Test MAE={test_metrics['MAE']:.4f}  "
          f"SAL={test_metrics['sal_eval_score']:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("  TFT — 2-Phase (Quantile + SAL) Walk-Forward CV (Keras)")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}  HIDDEN={HIDDEN}  N_HEADS={N_HEADS}")
    print(f"  QUANTILES={QUANTILES}  DROPOUT={DROPOUT}")
    print(f"  N_PAST={N_PAST}  N_KNOWN={N_KNOWN}  BATCH={BATCH_SIZE}")
    print()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default=None,
                        help="단일 phase만 실행 (quantile|sal_conservative|"
                             "sal_balanced|sal_aggressive). 생략 시 4개 전부.")
    args = parser.parse_args()

    build_tft("quantile").summary()
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
        if not FORCE and result_exists("tft", phase_tag):
            print(f"  ⏭️  {phase_tag} 결과 존재 — 건너뜀 (재학습: FORCE_RETRAIN=1)")
            continue
        fr, tm = run_tft_walk_forward(
            output_mode=output_mode, loss_fn=make_loss(), phase_tag=phase_tag,
        )
        _summary(phase_tag, fr, tm)
