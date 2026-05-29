"""
run_all.py — 전체 모델 순차 학습 + 결과 비교
==============================================
BiLSTM → iTransformer → TFT 순서로 학습하고
최종 비교 테이블 + 비교 그래프를 출력.

사용법:
    # 전체 학습 + 비교
    python run_all.py

    # 특정 모델만 학습
    python run_all.py --models bilstm itransformer

    # 학습 건너뛰고 기존 결과만 비교
    python run_all.py --skip-train

모델 개별 실행:
    python bilstm.py
    python itransformer.py
    python tft.py
"""
import os, json, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_MODELS = ["bilstm", "itransformer", "tft"]
SCRIPTS    = {m: f"{m}.py" for m in ALL_MODELS}
COLORS     = {"bilstm": "#2E75B6", "itransformer": "#C55A11", "tft": "#70AD47"}
LABELS     = {"bilstm": "Bi-LSTM", "itransformer": "iTransformer", "tft": "TFT"}


# ── 모델 실행 ─────────────────────────────────────────────────────────────────
def run_model(name: str) -> bool:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPTS[name])
    print(f"\n{'='*60}")
    print(f"  [{LABELS[name]}] 학습 시작")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script], check=False)
    ok = result.returncode == 0
    if not ok:
        print(f"  ⚠️  {name} 오류 발생 (code={result.returncode})")
    return ok


# ── 결과 로드 ─────────────────────────────────────────────────────────────────
def load_result(name: str):
    path = os.path.join(RESULTS_DIR, f"{name}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── 비교 테이블 출력 ──────────────────────────────────────────────────────────
def print_comparison_table(models):
    print("\n" + "=" * 74)
    print(f"  {'모델':<14} {'CV MAE':>8} {'CV RMSE':>9} {'CV MAPE':>9}"
          f" {'Test MAE':>9} {'Test RMSE':>10} {'Test MAPE':>10}")
    print("=" * 74)
    for name in models:
        r = load_result(name)
        if r is None:
            print(f"  {LABELS[name]:<14} {'(결과 없음)':>57}")
            continue
        cv = r["cv_mean"]; te = r["test_metrics"]
        print(f"  {LABELS[name]:<14} {cv['MAE']:>8.4f} {cv['RMSE']:>9.4f}"
              f" {cv['MAPE']:>8.2f}% {te['MAE']:>9.4f}"
              f" {te['RMSE']:>10.4f} {te['MAPE']:>9.2f}%")
    print("=" * 74)


# ── 비교 그래프 ───────────────────────────────────────────────────────────────
def plot_bar_comparison(models):
    results = {n: load_result(n) for n in models}
    results = {n: r for n, r in results.items() if r}
    if not results:
        return

    names    = list(results.keys())
    cv_mae   = [results[n]["cv_mean"]["MAE"]      for n in names]
    test_mae = [results[n]["test_metrics"]["MAE"]  for n in names]
    cv_mape  = [results[n]["cv_mean"]["MAPE"]      for n in names]
    test_mape= [results[n]["test_metrics"]["MAPE"] for n in names]
    labels   = [LABELS[n] for n in names]
    colors   = [COLORS[n] for n in names]

    x = np.arange(len(names)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, cv_vals, te_vals, ylabel, title in [
        (axes[0], cv_mae,  test_mae,  "MAE (normalized)", "MAE Comparison"),
        (axes[1], cv_mape, test_mape, "MAPE (%)",         "MAPE Comparison"),
    ]:
        b1 = ax.bar(x - w/2, cv_vals, w, label="CV Mean", color=colors, alpha=0.85)
        b2 = ax.bar(x + w/2, te_vals, w, label="Test",    color=colors, alpha=0.5, hatch="//")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(axis="y", alpha=0.3)
        fmt = ".4f" if "MAE" in ylabel else ".2f"
        suf = "" if "MAE" in ylabel else "%"
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                    f"{h:{fmt}}{suf}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Comparison — LLM Workload Prediction (BurstGPT)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  비교 그래프 저장: {out}")


def plot_fold_stability(models):
    """Fold별 MAE 선 그래프 (안정성 확인)."""
    results = {n: load_result(n) for n in models}
    results = {n: r for n, r in results.items() if r}
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    markers = ["o", "s", "^"]
    for i, (name, r) in enumerate(results.items()):
        folds    = [f["fold"] for f in r["fold_val_metrics"]]
        fold_mae = [f["MAE"]  for f in r["fold_val_metrics"]]
        ax.plot(folds, fold_mae, marker=markers[i], color=COLORS[name],
                label=LABELS[name], lw=1.5, markersize=7)
        ax.axhline(r["cv_mean"]["MAE"], color=COLORS[name],
                   ls="--", alpha=0.4, lw=1)

    ax.set_title("Fold-wise Validation MAE (Walk-Forward CV Stability)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Fold"); ax.set_ylabel("MAE (normalized)")
    ax.set_xticks(range(1, 6))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fold_stability.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Fold 안정성 그래프 저장: {out}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="학습 건너뛰고 기존 결과만 비교")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    args = parser.parse_args()

    if not args.skip_train:
        for name in args.models:
            run_model(name)

    print("\n\n" + "█" * 60)
    print("  최종 결과 비교")
    print("█" * 60)
    print_comparison_table(args.models)
    plot_bar_comparison(args.models)
    plot_fold_stability(args.models)
    print("\n  모든 결과: results/ 디렉토리 확인")
