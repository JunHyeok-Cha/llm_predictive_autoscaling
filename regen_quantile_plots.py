"""
저장된 quantile 모델(*_quantile_final.keras)을 로드해
  1) P10-P50-P90 밴드 예측 플롯 재생성
  2) 평가 감사: 분위 교차(crossing) 여부, 전역 vs 스파이크 구간 위반율
출력: results/analysis/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import utils
from utils import load_test, make_sequences, make_tft_sequences, RESULTS_DIR
from bilstm import build_bilstm, SEQ_LEN as SL_BILSTM
from itransformer import build_itransformer, SEQ_LEN as SL_ITR
from tft import build_tft, SEQ_LEN as SL_TFT

for cand in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic"]:
    if any(cand in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = cand
        break
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(OUT, exist_ok=True)
SPIKE_THR = 0.10   # target>0.10 = 상위 ~10.7% 고부하(스파이크) 구간

test_df = load_test()

CFG = {
    "bilstm":       (build_bilstm,       SL_BILSTM, "seq"),
    "itransformer": (build_itransformer, SL_ITR,    "seq"),
    "tft":          (build_tft,          SL_TFT,    "tft"),
}

audit_rows = []
for name, (build_fn, seq_len, kind) in CFG.items():
    wpath = os.path.join(RESULTS_DIR, f"{name}_quantile_final.keras")
    if not os.path.exists(wpath):
        print(f"  ⚠️  {wpath} 없음 — 건너뜀")
        continue
    print(f"\n[{name}] 모델 재구성 + 가중치 로드 (seq_len={seq_len}) ...")
    model = build_fn("quantile")
    model.load_weights(wpath)

    if kind == "tft":
        past, fut, y = make_tft_sequences(test_df, seq_len)
        preds = model.predict([past, fut], verbose=0, batch_size=512)
    else:
        X, y = make_sequences(test_df, seq_len)
        preds = model.predict(X, verbose=0, batch_size=512)

    preds = np.asarray(preds)
    p10, p50, p90 = preds[:, 0], preds[:, 1], preds[:, 2]
    y = np.asarray(y).reshape(-1)

    # ---- 감사 1: 분위 교차(quantile crossing) ----
    cross_10_50 = float(np.mean(p10 > p50))
    cross_50_90 = float(np.mean(p50 > p90))
    neg_band    = float(np.mean((p90 - p10) < 0))

    # ---- 감사 2: 전역 vs 스파이크 위반율 (decision=P90) ----
    viol_global = float(np.mean(p90 < y))
    spike = y > SPIKE_THR
    calm  = ~spike
    viol_spike = float(np.mean(p90[spike] < y[spike])) if spike.any() else float("nan")
    viol_calm  = float(np.mean(p90[calm]  < y[calm]))  if calm.any()  else float("nan")
    # 위반의 대부분이 스파이크에서 나는지: 위반 샘플 중 스파이크 비중
    viol_mask = p90 < y
    share_viol_in_spike = float(np.mean(spike[viol_mask])) if viol_mask.any() else float("nan")

    audit_rows.append({
        "model": name, "n": len(y),
        "spike_ratio": float(np.mean(spike)),
        "cross_P10>P50": cross_10_50, "cross_P50>P90": cross_50_90, "neg_band": neg_band,
        "viol_global": viol_global, "viol_calm": viol_calm, "viol_spike": viol_spike,
        "share_viol_in_spike": share_viol_in_spike,
        "band_width": float(np.mean(p90 - p10)),
    })

    # ---- 밴드 플롯 (앞 2000 스텝) ----
    n = min(2000, len(y))
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.fill_between(xs, p10[:n], p90[:n], color="orange", alpha=0.25,
                    label="P10–P90 밴드")
    ax.plot(xs, y[:n],    color="#1f77b4", lw=0.9, alpha=0.85, label="실제(Actual)")
    ax.plot(xs, p50[:n],  color="#d62728", lw=0.9, alpha=0.9,  label="예측 P50")
    ax.plot(xs, p90[:n],  color="#ff7f0e", lw=0.7, alpha=0.7,  label="용량결정 P90")
    # 스파이크 구간 위반 지점 표시
    vmask = (p90[:n] < y[:n])
    ax.scatter(xs[vmask], y[:n][vmask], s=10, color="red", zorder=5,
               label=f"SLA 위반({vmask.sum()}건)")
    ax.set_title(f"{name}_quantile — 테스트 예측 (P10–P90 밴드, 앞 {n} 스텝)\n"
                 f"전역 위반율={viol_global*100:.1f}%  /  스파이크 구간 위반율={viol_spike*100:.1f}%",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time step (min)")
    ax.set_ylabel("req_count (normalized)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(OUT, f"band_{name}_quantile_test_pred.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  저장: {p}")

# ---- 감사 요약 출력 ----
print("\n" + "=" * 92)
print("  평가 감사 요약 (quantile, 테스트셋)")
print("=" * 92)
hdr = ("model", "spike%", "cross10>50", "cross50>90", "negBand",
       "viol_glob", "viol_calm", "viol_spk", "viol중spk비중")
print(("{:<13}" + "{:>11}" * 8).format(*hdr))
for r in audit_rows:
    print(("{:<13}" + "{:>11.1%}" + "{:>11.2%}" * 3 + "{:>11.1%}" * 4).format(
        r["model"], r["spike_ratio"], r["cross_P10>P50"], r["cross_P50>P90"],
        r["neg_band"], r["viol_global"], r["viol_calm"], r["viol_spike"],
        r["share_viol_in_spike"]))
print("=" * 92)
import json
with open(os.path.join(OUT, "quantile_audit.json"), "w") as f:
    json.dump(audit_rows, f, indent=2)
print("저장:", os.path.join(OUT, "quantile_audit.json"))
