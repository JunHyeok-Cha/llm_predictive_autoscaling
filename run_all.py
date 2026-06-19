"""
run_all.py — 전체 모델 순차 학습 + 결과 비교 (2-Phase: Quantile + SAL)
=======================================================================
각 모델은 파일당 4개 서브실험을 순차 실행한다:
  Phase 1 — quantile (P10/P50/P90, pinball loss)
  Phase 2 — sal_conservative / sal_balanced / sal_aggressive (point, SAL loss)

따라서 모델당 결과 파일 4개 × 3개 모델 = 12개 설정을 모두 읽어 비교한다.
모든 설정은 공통 기준(SAL_SCENARIOS['balanced'])의 sal_eval_score로 동일하게 비교된다.

사용법:
    # 전체 학습 + 비교
    python run_all.py

    # 특정 모델만 학습
    python run_all.py --models bilstm itransformer

    # 학습 건너뛰고 기존 결과만 비교
    python run_all.py --skip-train

    # 특정 Phase만 비교 (표/그래프 한정)
    python run_all.py --skip-train --phases quantile sal_balanced

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
ALL_PHASES = ["quantile", "sal_conservative", "sal_balanced", "sal_aggressive"]
SCRIPTS    = {m: f"{m}.py" for m in ALL_MODELS}
LABELS     = {"bilstm": "Bi-LSTM", "itransformer": "iTransformer", "tft": "TFT"}
PHASE_LABELS = {
    "quantile":         "Quantile",
    "sal_conservative": "SAL-Conserv",
    "sal_balanced":     "SAL-Balanced",
    "sal_aggressive":   "SAL-Aggr",
}
# Phase별 색상 (모델 그룹 안에서 4개 바 구분)
PHASE_COLORS = {
    "quantile":         "#7F7F7F",
    "sal_conservative": "#2E75B6",
    "sal_balanced":     "#C55A11",
    "sal_aggressive":   "#70AD47",
}


# ── 모델 실행 ─────────────────────────────────────────────────────────────────
def phase_done(model: str, phase: str) -> bool:
    return os.path.exists(os.path.join(RESULTS_DIR, f"{model}_{phase}_results.json"))


def run_phase(name: str, phase: str) -> bool:
    """phase 하나를 별도 프로세스로 실행.
    프로세스를 phase 단위로 분리해야 한 프로세스가 여러 fold를 누적하며
    host RAM을 잠식하다 OOM(code=-9)으로 죽는 걸 막을 수 있다 (TFT처럼).
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), SCRIPTS[name])
    force  = os.environ.get("FORCE_RETRAIN") == "1"
    if not force and phase_done(name, phase):
        print(f"  ⏭️  [{LABELS[name]}] {phase} 결과 존재 — 건너뜀")
        return True
    print(f"\n{'='*60}")
    print(f"  [{LABELS[name]}] {phase} 학습 (격리 프로세스)")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script, "--phase", phase], check=False)
    ok = result.returncode == 0
    if not ok:
        print(f"  ⚠️  {name}/{phase} 오류 (code={result.returncode})"
              + ("  ← -9는 OOM(시스템 RAM) kill" if result.returncode == -9 else ""))
    return ok


# ── 결과 로드 ─────────────────────────────────────────────────────────────────
def load_result(model: str, phase: str):
    path = os.path.join(RESULTS_DIR, f"{model}_{phase}_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


# ── 비교 테이블 출력 ──────────────────────────────────────────────────────────
def print_comparison_table(models, phases):
    cols = ["MAE", "RMSE", "SMAPE", "ViolRate", "AvgOver", "AvgUnder", "SAL"]
    header = (f"  {'Model':<13}{'Phase':<13}"
              + "".join(f"{c:>10}" for c in cols))
    print("\n" + "=" * len(header))
    print("  CV 평균 지표 (SAL = 공통기준 balanced sal_eval_score)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in models:
        for phase in phases:
            r = load_result(name, phase)
            if r is None:
                print(f"  {LABELS[name]:<13}{PHASE_LABELS[phase]:<13}{'(결과 없음)':>20}")
                continue
            cv = r["cv_mean"]
            print(f"  {LABELS[name]:<13}{PHASE_LABELS[phase]:<13}"
                  f"{cv['MAE']:>10.4f}{cv['RMSE']:>10.4f}{cv['SMAPE']:>9.2f}%"
                  f"{cv['violation_rate']:>10.3f}{cv['avg_overprovision']:>10.4f}"
                  f"{cv['avg_underprovision']:>10.4f}{cv['sal_eval_score']:>10.4f}")
    print("=" * len(header))

    # Test 동일 지표
    print("\n" + "=" * len(header))
    print("  Test 지표")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name in models:
        for phase in phases:
            r = load_result(name, phase)
            if r is None:
                continue
            te = r["test_metrics"]
            print(f"  {LABELS[name]:<13}{PHASE_LABELS[phase]:<13}"
                  f"{te['MAE']:>10.4f}{te['RMSE']:>10.4f}{te['SMAPE']:>9.2f}%"
                  f"{te['violation_rate']:>10.3f}{te['avg_overprovision']:>10.4f}"
                  f"{te['avg_underprovision']:>10.4f}{te['sal_eval_score']:>10.4f}")
    print("=" * len(header))


# ── 그룹 막대그래프 ───────────────────────────────────────────────────────────
def _grouped_bar(ax, models, phases, value_fn, title, ylabel, fmt=".4f", suf=""):
    """모델별로 그룹화 → 각 그룹 안에 Phase 바를 나란히."""
    n_ph = len(phases)
    x = np.arange(len(models))
    w = 0.8 / n_ph
    any_bar = False
    for j, phase in enumerate(phases):
        vals = []
        for name in models:
            r = load_result(name, phase)
            vals.append(value_fn(r) if r else np.nan)
        offs = x - 0.4 + w * (j + 0.5)
        bars = ax.bar(offs, vals, w, label=PHASE_LABELS[phase],
                      color=PHASE_COLORS[phase], alpha=0.9)
        any_bar = any_bar or np.any(~np.isnan(vals))
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:{fmt}}{suf}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in models])
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    return any_bar


def plot_comparison(models, phases, split="cv_mean"):
    """split: 'cv_mean' 또는 'test_metrics'."""
    panels = [
        ("MAE",            lambda r: r[split]["MAE"],               "MAE (normalized)",      ".4f", ""),
        ("SAL Eval Score", lambda r: r[split]["sal_eval_score"],    "SAL score (공통기준)",   ".4f", ""),
        ("Violation Rate", lambda r: r[split]["violation_rate"],    "Violation rate",        ".3f", ""),
        ("Avg Underprov.", lambda r: r[split]["avg_underprovision"],"Avg underprovision",    ".4f", ""),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    handles = labels = None
    for ax, (title, fn, ylabel, fmt, suf) in zip(axes.ravel(), panels):
        _grouped_bar(ax, models, phases, fn, title, ylabel, fmt, suf)
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(phases),
               bbox_to_anchor=(0.5, 1.0))
    tag = "CV" if split == "cv_mean" else "Test"
    plt.suptitle(f"2-Phase Comparison ({tag}) — Quantile vs SAL scenarios",
                 fontsize=14, fontweight="bold", y=1.04)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"phase_comparison_{split}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  비교 그래프 저장: {out}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="학습 건너뛰고 기존 결과만 비교")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument("--phases", nargs="+", default=ALL_PHASES,
                        choices=ALL_PHASES,
                        help="비교에 포함할 Phase (표/그래프 한정)")
    args = parser.parse_args()

    if not args.skip_train:
        # phase 단위 격리 프로세스로 실행 (모델당 4 phase) → 프로세스별 메모리 회수
        for name in args.models:
            for phase in args.phases:
                run_phase(name, phase)

    print("\n\n" + "█" * 60)
    print("  최종 결과 비교 (2-Phase)")
    print("█" * 60)
    print_comparison_table(args.models, args.phases)
    plot_comparison(args.models, args.phases, split="cv_mean")
    plot_comparison(args.models, args.phases, split="test_metrics")
    print("\n  모든 결과: results/ 디렉토리 확인")
