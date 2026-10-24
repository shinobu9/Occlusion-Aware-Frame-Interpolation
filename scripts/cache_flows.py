"""Cache frozen RAFT flows and common midpoint features, with disjoint GPU shards."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from src.core import make_warps, make_features, dis_region, occ_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    if args.init:
        for s in ("train", "val", "test"):
            n = len(np.load(args.root / f"{s}_frames.npy", mmap_mode="r"))
            for suffix, shape, dtype in (("flows", (n, 2, 2, 256, 448), np.float16),
                                         ("features", (n, 18, 256, 448), np.float16),
                                         ("regions", (n, 2, 256, 448), np.uint8)):
                p = args.root / f"{s}_{suffix}.npy"
                if not p.exists():
                    np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape).flush()
        print("Cache arrays initialized", flush=True)
        return
    torch.set_num_threads(4)
    device = torch.device("cuda")
    weights = Raft_Small_Weights.C_T_V2
    model = raft_small(weights=weights, progress=False).to(device).eval()
    pre = weights.transforms()
    start = time.monotonic()
    total = 0
    with torch.inference_mode():
        for s in ("train", "val", "test"):
            marker = args.root / f"{s}_cache_rank{args.rank}.json"
            if marker.exists():
                print(f"already cached {s} rank {args.rank}", flush=True)
                continue
            frames = np.load(args.root / f"{s}_frames.npy", mmap_mode="r")
            flows = np.load(args.root / f"{s}_flows.npy", mmap_mode="r+")
            features = np.load(args.root / f"{s}_features.npy", mmap_mode="r+")
            regions = np.load(args.root / f"{s}_regions.npy", mmap_mode="r+")
            indices = np.array_split(np.arange(len(frames)), args.world)[args.rank]
            for j in range(0, len(indices), args.batch):
                ids = indices[j:j+args.batch]
                f = torch.from_numpy(np.array(frames[ids])).to(device).float() / 255
                a, b = f[:, 0], f[:, 2]
                aa, bb = pre(a, b)
                # Quantize once so training and evaluation use identical flow precision.
                f01 = model(aa, bb)[-1].half().float()
                f10 = model(bb, aa)[-1].half().float()
                wd = make_warps(a, b, f01, f10)
                feat = make_features(wd)
                assert torch.isfinite(feat).all()
                flows[ids] = torch.stack((f01, f10), 1).cpu().numpy()
                features[ids] = feat.half().cpu().numpy()
                regions[ids] = torch.cat((dis_region(wd), occ_region(wd)), 1).byte().cpu().numpy()
                total += len(ids)
                if j % (args.batch * 10) == 0 or j + args.batch >= len(indices):
                    elapsed = time.monotonic()-start
                    print(f"rank {args.rank} {s} {min(j+args.batch,len(indices))}/{len(indices)}; total {total}, {elapsed:.1f}s, {total/elapsed:.1f} triplets/s", flush=True)
            for arr in (flows, features, regions):
                arr.flush()
            marker.write_text(json.dumps(dict(rank=args.rank, world=args.world, count=len(indices),
                                             first=int(indices[0]), last=int(indices[-1]),
                                             weights="Raft_Small_Weights.C_T_V2", precision="flow fp16, splat fp32, features fp16")))
    print(f"DONE rank {args.rank}, {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
