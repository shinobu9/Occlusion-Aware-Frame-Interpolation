"""Post-hoc diagnostic: does the mask gain persist after removing image borders?"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.core import MaskUNet, heuristic_mask, blend
from src.data import CachedData, wd_from_features
from src.metrics import measure, summarize
from src.runtime import setup
from scripts.report_masks import paired_ci

setup(42)
root = Path(__file__).resolve().parents[1]
data = CachedData(root / "data", "test")
model = MaskUNet().cuda().eval()
model.load_state_dict(torch.load(root / "runs/mask_s42/best.pt", map_location="cuda", weights_only=False)["model"])
strength = json.loads((root / "artifacts/heuristic_selection.json").read_text())["strength"]
rows = []
with torch.inference_mode():
    for start in range(0, len(data), 8):
        ids = list(range(start, min(start+8, len(data))))
        f, target, reg = data.batch(ids)
        wd = wd_from_features(f)
        for name, pred in [("fixed", blend(wd, .5)), ("heuristic", blend(wd, heuristic_mask(wd, strength))),
                           ("learned", blend(wd, model(f)))]:
            values = measure(pred[..., 16:-16, 16:-16], target[..., 16:-16, 16:-16], {})
            for i, idx in enumerate(ids):
                rows.append(dict(key=data.manifest[idx]["key"], method=name,
                                 **{k: float(v[i]) for k, v in values.items()}))
df = pd.DataFrame(rows)
df.to_csv(root / "artifacts/mask_interior_per_triplet.csv", index=False)
pd.DataFrame(summarize(df)).to_csv(root / "artifacts/mask_interior_summary.csv", index=False)
rng = np.random.default_rng(42)
intervals = []
for metric in ("psnr", "ssim", "mae"):
    pivot = df.pivot(index="key", columns="method", values=metric)
    for baseline in ("fixed", "heuristic"):
        delta, low, high = paired_ci((pivot.learned-pivot[baseline]).to_numpy(), rng)
        intervals.append(dict(metric=metric, baseline=baseline, delta=delta, ci_lo=low, ci_hi=high))
pd.DataFrame(intervals).to_csv(root / "artifacts/mask_interior_ci.csv", index=False)
print(pd.DataFrame(summarize(df)).to_string(index=False))
print(pd.DataFrame(intervals).to_string(index=False))
