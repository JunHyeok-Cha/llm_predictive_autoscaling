"""
results/ 폴더의 결과 JSON을 읽어 보고서용 비교차트와 종합 분석표를 생성한다.
산출물: results/analysis/ 에 PNG 차트 + CSV/MD 표
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# ---- 한글 폰트 ----
for cand in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(__file__).parent / "results"
OUT = RESULTS / "analysis"
OUT.mkdir(exist_ok=True)

MODELS = ["bilstm", "itransformer", "tft"]
STRATS = ["quantile", "sal_aggressive", "sal_balanced", "sal_conservative"]
STRAT_LABEL = {
    "quantile": "Quantile",
    "sal_aggressive": "SAL-Aggressive",
    "sal_balanced": "SAL-Balanced",
    "sal_conservative": "SAL-Conservative",
}
MODEL_LABEL = {"bilstm": "BiLSTM", "itransformer": "iTransformer", "tft": "TFT"}

# ---- 데이터 로드 ----
rows = []
for m in MODELS:
    for s in STRATS:
        fp = RESULTS / f"{m}_{s}_results.json"
        if not fp.exists():
            continue
        d = json.loads(fp.read_text())
        t = d["test_metrics"]
        c = d["cv_mean"]
        rows.append({
            "model": m, "strategy": s,
            "MAE": t["MAE"], "RMSE": t["RMSE"], "SMAPE": t["SMAPE"],
            "violation_rate": t["violation_rate"],
            "violation_rate_peak": t.get("violation_rate_peak", np.nan),
            "avg_overprovision": t["avg_overprovision"],
            "avg_underprovision": t["avg_underprovision"],
            "sal_eval_score": t["sal_eval_score"],
            "band_width": t.get("band_width", np.nan),
            "cv_MAE": c["MAE"], "cv_violation": c["violation_rate"],
        })
df = pd.DataFrame(rows)

# ===================================================================
# 1) 전략별 그룹 막대그래프 (테스트셋 핵심 지표)
# ===================================================================
metrics = [
    ("MAE", "MAE (낮을수록 정확)", False),
    ("RMSE", "RMSE (낮을수록 정확)", False),
    ("violation_rate", "Violation Rate (SLA 위반, 낮을수록 안전)", False),
    ("avg_overprovision", "Over-provisioning (비용, 낮을수록 효율)", False),
    ("sal_eval_score", "SAL Eval Score (종합 목적치, 낮을수록 우수)", False),
    ("SMAPE", "SMAPE (%)", False),
]
colors = {"bilstm": "#4C72B0", "itransformer": "#DD8452", "tft": "#55A868"}
x = np.arange(len(STRATS))
w = 0.26

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (key, title, _) in zip(axes.ravel(), metrics):
    for i, m in enumerate(MODELS):
        sub = df[df.model == m].set_index("strategy").reindex(STRATS)
        ax.bar(x + (i - 1) * w, sub[key].values, w,
               label=MODEL_LABEL[m], color=colors[m])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=15, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("모델 × 전략 테스트셋 성능 비교", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "01_metric_comparison.png", dpi=150)
plt.close(fig)

# ===================================================================
# 2) 비용-위험 트레이드오프 산점도 (overprovision vs violation_rate)
# ===================================================================
markers = {"quantile": "o", "sal_aggressive": "^",
           "sal_balanced": "s", "sal_conservative": "D"}
fig, ax = plt.subplots(figsize=(10, 7.5))
for _, r in df.iterrows():
    ax.scatter(r["avg_overprovision"], r["violation_rate"],
               s=260, color=colors[r["model"]], marker=markers[r["strategy"]],
               edgecolor="black", linewidth=0.8, alpha=0.85, zorder=3)
    ax.annotate(f"{MODEL_LABEL[r['model']]}\n{STRAT_LABEL[r['strategy']]}",
                (r["avg_overprovision"], r["violation_rate"]),
                fontsize=7, ha="center", va="center",
                xytext=(0, 22), textcoords="offset points")
ax.set_xlabel("Over-provisioning (비용 ↑) →", fontsize=12)
ax.set_ylabel("Violation Rate (SLA 위반 위험 ↑) →", fontsize=12)
ax.set_title("비용 vs SLA 위험 트레이드오프\n(좌하단 = 저비용·저위험 = 이상적)",
             fontsize=14, fontweight="bold")
ax.grid(alpha=0.3)
# 모델 색상 범례
from matplotlib.lines import Line2D
mh = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m],
             markersize=12, label=MODEL_LABEL[m]) for m in MODELS]
sh = [Line2D([0], [0], marker=markers[s], color="w", markerfacecolor="gray",
             markeredgecolor="black", markersize=11, label=STRAT_LABEL[s]) for s in STRATS]
ax.legend(handles=mh + sh, fontsize=9, loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "02_cost_risk_tradeoff.png", dpi=150)
plt.close(fig)

# ===================================================================
# 3) CV 폴드별 안정성 (sal_eval_score, 폴드별 라인)
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, m in zip(axes, MODELS):
    for s in STRATS:
        fp = RESULTS / f"{m}_{s}_results.json"
        d = json.loads(fp.read_text())
        folds = [fm_["fold"] for fm_ in d["fold_val_metrics"]]
        vals = [fm_["sal_eval_score"] for fm_ in d["fold_val_metrics"]]
        ax.plot(folds, vals, marker="o", label=STRAT_LABEL[s])
    ax.set_title(f"{MODEL_LABEL[m]} — 폴드별 SAL Score", fontweight="bold")
    ax.set_xlabel("CV Fold")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("sal_eval_score (val)")
fig.suptitle("교차검증 폴드별 안정성 (Fold 2는 부하≈0 구간으로 이상치)",
             fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "03_cv_fold_stability.png", dpi=150)
plt.close(fig)

# ===================================================================
# 4) 종합 분석표 (CSV + 렌더링 이미지 + Markdown)
# ===================================================================
tbl = df.copy()
tbl["model"] = tbl["model"].map(MODEL_LABEL)
tbl["strategy"] = tbl["strategy"].map(STRAT_LABEL)
disp = tbl[["model", "strategy", "MAE", "RMSE", "SMAPE", "violation_rate",
            "violation_rate_peak", "avg_overprovision", "avg_underprovision",
            "sal_eval_score", "band_width"]]
disp = disp.rename(columns={
    "model": "모델", "strategy": "전략", "violation_rate": "위반율(전역)",
    "violation_rate_peak": "위반율(피크)",
    "avg_overprovision": "과프로비저닝", "avg_underprovision": "언더프로비저닝",
    "sal_eval_score": "SAL점수", "band_width": "밴드폭",
})
disp.to_csv(OUT / "summary_test_metrics.csv", index=False, encoding="utf-8-sig")

# Markdown
fmt = disp.copy()
for col in fmt.columns:
    if col in ("모델", "전략"):
        continue
    if col == "SMAPE":
        fmt[col] = fmt[col].map(lambda v: f"{v:.1f}")
    else:
        fmt[col] = fmt[col].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
(OUT / "summary_test_metrics.md").write_text(fmt.to_markdown(index=False), encoding="utf-8")

# 표 이미지 (지표별 최우수 셀 강조)
fig, ax = plt.subplots(figsize=(15, 5.2))
ax.axis("off")
cell = fmt.values
table = ax.table(cellText=cell, colLabels=fmt.columns,
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
# 헤더 스타일
for j in range(len(fmt.columns)):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")
# 최우수(최소값) 강조 — 낮을수록 좋은 지표들
lower_better = ["MAE", "RMSE", "위반율(전역)", "위반율(피크)", "과프로비저닝", "SAL점수"]
for col in lower_better:
    cidx = list(fmt.columns).index(col)
    vals = pd.to_numeric(disp[col], errors="coerce")
    best = vals.idxmin()
    table[best + 1, cidx].set_facecolor("#A9DFBF")
# 모델별 행 음영
for i in range(len(disp)):
    if (i // 4) % 2 == 1:
        for j in range(len(fmt.columns)):
            if table[i + 1, j].get_facecolor()[:3] == (1.0, 1.0, 1.0):
                table[i + 1, j].set_facecolor("#F2F3F4")
ax.set_title("종합 분석표 — 테스트셋 (초록=지표별 최우수)",
             fontsize=14, fontweight="bold", pad=12)
fig.tight_layout()
fig.savefig(OUT / "04_summary_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# 5) 전역 vs 피크(상위10% 부하) 위반율 — 착시 폭로 차트
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
x5 = np.arange(len(STRATS))
for ax, m in zip(axes, MODELS):
    sub = df[df.model == m].set_index("strategy").reindex(STRATS)
    ax.bar(x5 - 0.2, sub["violation_rate"].values, 0.4,
           label="전역 위반율", color="#95A5A6")
    ax.bar(x5 + 0.2, sub["violation_rate_peak"].values, 0.4,
           label="피크 위반율(상위10% 부하)", color="#C0392B")
    for i, (g, p) in enumerate(zip(sub["violation_rate"], sub["violation_rate_peak"])):
        ax.annotate(f"×{p/g:.1f}" if g > 0 else "", (i, p), ha="center",
                    va="bottom", fontsize=8, color="#C0392B")
    ax.set_title(MODEL_LABEL[m], fontweight="bold")
    ax.set_xticks(x5)
    ax.set_xticklabels([STRAT_LABEL[s] for s in STRATS], rotation=15, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("SLA 위반율")
fig.suptitle("전역 위반율 vs 피크(고부하) 위반율 — 전역값은 calm 구간 지배로 과소평가\n"
             "(×N = 피크가 전역의 N배)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "05_global_vs_peak_violation.png", dpi=150)
plt.close(fig)

print("생성 완료 →", OUT)
for f in sorted(OUT.iterdir()):
    print("  -", f.name)
print("\n[종합표 미리보기]")
print(fmt.to_string(index=False))
