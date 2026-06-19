"""
저장된 *_final.keras(추론 전용, 재학습 X)로 테스트셋을 다시 예측해
기존 결과 JSON의 test_metrics에 violation_rate_peak를 백필한다.
- fold_val_metrics/cv_mean은 폴드별 모델이 저장돼 있지 않아 건드리지 않음.
- 재추론한 전역 violation_rate/MAE가 기존 JSON과 일치하는지 검증 출력.
"""
import os, json
import numpy as np

import utils
from utils import (load_test, make_sequences, make_tft_sequences, RESULTS_DIR,
                   peak_violation_rate)
from bilstm import build_bilstm, SEQ_LEN as SL_BILSTM
from itransformer import build_itransformer, SEQ_LEN as SL_ITR
from tft import build_tft, SEQ_LEN as SL_TFT

CFG = {
    "bilstm":       (build_bilstm,       SL_BILSTM, "seq"),
    "itransformer": (build_itransformer, SL_ITR,    "seq"),
    "tft":          (build_tft,          SL_TFT,    "tft"),
}
PHASES = ["quantile", "sal_conservative", "sal_balanced", "sal_aggressive"]

test_df = load_test()
# 모델·phase별로 동일 seq_len을 공유하므로 시퀀스는 모델당 1회만 생성해 캐시
seq_cache = {}

print(f"{'model_phase':<32}{'MAE(old→new)':>22}{'viol(old→new)':>22}{'viol_peak':>11}")
print("-" * 88)
for name, (build_fn, seq_len, kind) in CFG.items():
    if kind == "tft":
        past, fut, y = make_tft_sequences(test_df, seq_len)
        Xin = [past, fut]
    else:
        X, y = make_sequences(test_df, seq_len)
        Xin = X
    y = np.asarray(y).reshape(-1)

    for phase in PHASES:
        tag = f"{name}_{phase}"
        jpath = os.path.join(RESULTS_DIR, f"{tag}_results.json")
        wpath = os.path.join(RESULTS_DIR, f"{tag}_final.keras")
        if not (os.path.exists(jpath) and os.path.exists(wpath)):
            print(f"{tag:<32}  (파일 없음 — 건너뜀)")
            continue

        output_mode = "quantile" if phase == "quantile" else "point"
        model = build_fn(output_mode)
        model.load_weights(wpath)
        preds = model.predict(Xin, verbose=0, batch_size=512)
        preds = np.asarray(preds)

        if output_mode == "quantile":
            decision = preds[:, 2]   # P90
        else:
            decision = preds.reshape(-1)

        new_mae  = float(np.mean(np.abs(y - (preds[:, 1] if output_mode == "quantile" else decision))))
        new_viol = float(np.mean(decision < y))
        new_peak = peak_violation_rate(y, decision)

        with open(jpath) as f:
            res = json.load(f)
        tm = res["test_metrics"]
        old_mae, old_viol = tm.get("MAE"), tm.get("violation_rate")

        # 키 순서 유지: violation_rate 바로 뒤에 violation_rate_peak 삽입
        new_tm = {}
        for k, v in tm.items():
            new_tm[k] = v
            if k == "violation_rate":
                new_tm["violation_rate_peak"] = new_peak
        if "violation_rate_peak" not in new_tm:   # 안전장치
            new_tm["violation_rate_peak"] = new_peak
        res["test_metrics"] = new_tm
        with open(jpath, "w") as f:
            json.dump(res, f, indent=2)

        flag = "" if abs(new_viol - old_viol) < 5e-3 else "  ⚠️불일치"
        print(f"{tag:<32}{old_mae:>11.4f}→{new_mae:<10.4f}"
              f"{old_viol:>11.4f}→{new_viol:<10.4f}{new_peak:>11.4f}{flag}")

print("-" * 88)
print("백필 완료. (fold_val_metrics/cv_mean은 폴드 모델 미저장으로 미반영)")
