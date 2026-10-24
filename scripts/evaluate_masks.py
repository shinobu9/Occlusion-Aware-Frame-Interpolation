"""Select heuristic on validation or evaluate all mask seeds on a fixed test shard."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.core import MaskUNet, blend, heuristic_mask, oracle_alpha, backward_warp
from src.data import CachedData, wd_from_features
from src.metrics import measure, boundary_region, summarize
from src.runtime import setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("artifacts"))
    ap.add_argument("--models", type=Path, nargs="*", default=[])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()
    setup(42)
    args.out.mkdir(parents=True, exist_ok=True)
    data = CachedData(args.data, args.split)
    models = {}
    for p in args.models:
        model = MaskUNet().cuda().eval()
        model.load_state_dict(torch.load(p / "best.pt", map_location="cuda", weights_only=False)["model"])
        models[p.name] = model
    if args.split == "test":
        strength = json.loads((args.out / "heuristic_selection.json").read_text())["strength"]
        strengths = [strength]
    else:
        strengths = [.5, 1., 2.]
    indices = np.array_split(np.arange(len(data)), args.world)[args.rank]
    raw_flows = np.load(args.data / f"{args.split}_flows.npy", mmap_mode="r")
    rows = []
    start = time.monotonic()
    with torch.inference_mode():
        for j in range(0, len(indices), 8):
            ids = indices[j:j+8]
            f, t, reg = data.batch(ids)
            wd = wd_from_features(f)
            regions = dict(full=torch.ones_like(reg[:, :1]), dis=reg[:, :1], occ=reg[:, 1:2], boundary=boundary_region(f))
            frames = torch.from_numpy(np.array(data.frames[ids])).cuda().float()/255
            flow = torch.from_numpy(np.array(raw_flows[ids])).cuda().float()
            preds = dict(fixed=blend(wd, .5), oracle=blend(wd, oracle_alpha(wd, t)),
                         old_half_flow=.5*backward_warp(frames[:, 0], -.5*flow[:, 0])+.5*backward_warp(frames[:, 2], -.5*flow[:, 1]),
                         unwarped=.5*(frames[:, 0]+frames[:, 2]))
            preds.update({f"heuristic_{s:g}": blend(wd, heuristic_mask(wd, s)) for s in strengths})
            for name, model in models.items():
                preds[name] = blend(wd, model(f))
            for name, pred in preds.items():
                metrics = measure(pred, t, regions)
                for k, idx in enumerate(ids):
                    rows.append(dict(key=data.manifest[idx]["key"], index=int(idx), method=name,
                                     **{m: float(v[k]) for m, v in metrics.items()}))
            if j % 80 == 0:
                print(f"{args.split} rank {args.rank}: {j+len(ids)}/{len(indices)}, {time.monotonic()-start:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"masks_{args.split}_rank{args.rank}.csv", index=False)
    summary = summarize(df)
    if args.split == "val":
        best = max((r for r in summary if r["method"].startswith("heuristic_")), key=lambda r: r["pooled_psnr_dis"])
        (args.out / "heuristic_selection.json").write_text(json.dumps(dict(strength=float(best["method"].split("_")[1]),
                                                                        selected_on="validation pooled PSNR dis", results=summary), indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
