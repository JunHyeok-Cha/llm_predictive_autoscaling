"""
utils.py — 공통 유틸리티 (Keras/TensorFlow)
  - 데이터 로드
  - 슬라이딩 윈도우 배열 생성
  - 평가 지표 (MAE, RMSE, MAPE)
  - 결과 저장 / 시각화
"""
import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    일반 모델용 (BiLSTM / iTransformer).
    X: (N, seq_len, n_features)
    y: (N,)
    """
    X_arr = df[FEATURE_COLS].values.astype(np.float32)
    y_arr = df["target"].values.astype(np.float32)

    X_list, y_list = [], []
    for i in range(seq_len, len(X_arr)):
        if np.isnan(y_arr[i]) or np.isnan(X_arr[i]).any():
            continue
        X_list.append(X_arr[i - seq_len:i])
        y_list.append(y_arr[i])

    return np.array(X_list), np.array(y_list)


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

    past_list, future_list, y_list = [], [], []
    for i in range(seq_len, len(y_arr) - PRED_HORIZON):
        if np.isnan(y_arr[i]) or np.isnan(past_arr[i]).any():
            continue
        past_list.append(past_arr[i - seq_len:i])
        future_list.append(future_arr[i:i + PRED_HORIZON])
        y_list.append(y_arr[i])

    return (np.array(past_list),
            np.array(future_list),
            np.array(y_list))


# ── 평가 지표 ─────────────────────────────────────────────────────────────────
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps=1e-6):
    mask = np.abs(y_true) > eps
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask])
                                / y_true[mask])) * 100)

def compute_metrics(y_true, y_pred):
    return {"MAE":  mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred)}


# ── 결과 저장 ─────────────────────────────────────────────────────────────────
def save_loss_curve(history, fold: int, model_name: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(history.history["loss"],     label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title(f"{model_name} — Fold {fold} Loss Curve")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE Loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_fold{fold}_loss.png")
    plt.savefig(path, dpi=120); plt.close()

def save_pred_plot(y_true, y_pred, model_name: str, n: int = 2000):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(y_true[:n], label="Actual",    alpha=0.8, lw=0.9)
    ax.plot(y_pred[:n], label="Predicted", alpha=0.8, lw=0.9)
    ax.set_title(f"{model_name} — Final Test Predictions (first {n} steps)")
    ax.set_xlabel("Time step (min)"); ax.set_ylabel("req_count (normalized)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name}_test_pred.png")
    plt.savefig(path, dpi=120); plt.close()
    print(f"  예측 그래프 저장: {path}")

def save_results(fold_results: list, test_metrics: dict, model_name: str):
    summary = {
        "model": model_name,
        "fold_val_metrics": fold_results,
        "cv_mean": {
            "MAE":  float(np.mean([r["MAE"]  for r in fold_results])),
            "RMSE": float(np.mean([r["RMSE"] for r in fold_results])),
            "MAPE": float(np.mean([r["MAPE"] for r in fold_results])),
        },
        "test_metrics": test_metrics,
    }
    path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  결과 저장: {path}")
    return summary


# ── Walk-Forward CV 공통 실행기 (BiLSTM / iTransformer 용) ───────────────────
def run_walk_forward(
    build_fn,          # 모델을 반환하는 함수 build_fn() → keras.Model
    seq_len: int,
    model_name: str,
    batch_size: int = 256,
    max_epochs: int = 60,
    patience:   int = 8,
):
    """
    build_fn(): 호출할 때마다 새 Keras 모델을 반환해야 함.
    """
    import tensorflow as tf
    from tensorflow import keras

    fold_results = []
    last_model   = None

    for fold in range(1, N_FOLDS + 1):
        print(f"\n── Fold {fold}/{N_FOLDS} ──────────────────")
        tr_df, vl_df = load_fold(fold)
        X_tr, y_tr   = make_sequences(tr_df, seq_len)
        X_vl, y_vl   = make_sequences(vl_df, seq_len)
        print(f"  train: {X_tr.shape}  val: {X_vl.shape}")

        model = build_fn()
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mae",
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience,
                restore_best_weights=True, verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0,
            ),
        ]

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        vl_preds = model.predict(X_vl, verbose=0).squeeze()
        metrics  = compute_metrics(y_vl, vl_preds)
        metrics["fold"] = fold
        fold_results.append(metrics)
        print(f"  Fold {fold} → MAE={metrics['MAE']:.4f}  "
              f"RMSE={metrics['RMSE']:.4f}  MAPE={metrics['MAPE']:.2f}%")

        save_loss_curve(history, fold, model_name)
        last_model = model

    # 최종 테스트
    print("\n── Final Test ──────────────────────────────")
    test_df    = load_test()
    X_te, y_te = make_sequences(test_df, seq_len)
    te_preds   = last_model.predict(X_te, verbose=0).squeeze()
    test_metrics = compute_metrics(y_te, te_preds)
    print(f"  Test → MAE={test_metrics['MAE']:.4f}  "
          f"RMSE={test_metrics['RMSE']:.4f}  MAPE={test_metrics['MAPE']:.2f}%")

    save_pred_plot(y_te, te_preds, model_name)
    save_results(fold_results, test_metrics, model_name)

    model_path = os.path.join(RESULTS_DIR, f"{model_name}_final.keras")
    last_model.save(model_path)
    print(f"  모델 저장: {model_path}")

    return fold_results, test_metrics
