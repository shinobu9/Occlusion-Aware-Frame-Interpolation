"""Deterministic per-triplet noise for validation sampler diagnostics and final testing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import lpips
from src.data import CachedData
from src.metrics import measure, boundary_region, summarize
from src.refiner import RefinerUNet, RES_SCALE, l1_predict, best_dis_window
from src.runtime import setup
from scripts.train_refiner import sample


def noise_for_keys(keys, sample_index=0, shape=(3, 256, 448)):
    noises = []
    for key in keys:
        seed = int(hashlib.sha256(f"evaluation42/{key}/{sample_index}".encode()).hexdigest()[:8], 16)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        noises.append(torch.randn(*shape, device="cuda", generator=gen))
    return torch.stack(noises)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--solver", choices=["euler", "heun"], default="euler")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.)
    ap.add_argument("--mean-samples", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--checkpoint", default="best.pt")
    args = ap.parse_args()
    setup(42)
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.run / "config.json").read_text())
    assert args.guidance == 1. or config["cfg_dropout"] > 0, "CFG needs condition-dropout training"
    model = RefinerUNet().cuda().eval()
    ck = torch.load(args.run / args.checkpoint, map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"])
    lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
    data = CachedData(args.data, args.split)
    cond = np.load(args.data / f"{args.split}_condition.npy", mmap_mode="r")
    indices = np.array_split(np.arange(min(args.limit or len(data), len(data))), args.world)[args.rank]
    rows = []
    start = time.monotonic()
    inference_seconds = 0.
    with torch.inference_mode():
        for j in range(0, len(indices), 4):
            ids = indices[j:j+4]
            keys = [data.manifest[i]["key"] for i in ids]
            f, t, reg = data.batch(ids)
            c = torch.from_numpy(np.array(cond[ids])).cuda().float()
            torch.cuda.synchronize()
            ti = time.monotonic()
            if config["mode"] == "l1":
                r = l1_predict(model, c)
            else:
                r = torch.stack([sample(model, c, args.steps, args.solver, guidance=args.guidance,
                                        noise=noise_for_keys(keys, k)) for k in range(args.mean_samples)]).mean(0)
            torch.cuda.synchronize()
            inference_seconds += time.monotonic()-ti
            pred = c[:, :3]+RES_SCALE*r
            regions = dict(full=torch.ones_like(reg[:, :1]), dis=reg[:, :1], occ=reg[:, 1:2], boundary=boundary_region(f))
            wins = best_dis_window(reg[:, :1])
            crop = lambda x: torch.cat([x[k:k+1, :, y:y+128, x0:x0+128] for k, (y, x0) in enumerate(wins)])
            for method, p in ((args.run.name, pred), (args.run.name+"_dis", c[:, :3]+RES_SCALE*r*reg[:, :1]), ("learned", c[:, :3])):
                metrics = measure(p, t, regions)
                pc = p.clamp(0, 1)
                metrics["lpips"] = lp(pc*2-1, t*2-1).flatten().cpu().numpy()
                metrics["lpips_window"] = lp(crop(pc)*2-1, crop(t)*2-1).flatten().cpu().numpy()
                for k, idx in enumerate(ids):
                    rows.append(dict(key=keys[k], index=int(idx), method=method, **{m: float(v[k]) for m, v in metrics.items()}))
            if j % 40 == 0:
                elapsed = time.monotonic()-start
                print(f"{args.run.name} {args.split} {j+len(ids)}/{len(indices)}; elapsed {elapsed:.1f}s; ETA {elapsed/(j+len(ids))*(len(indices)-j-len(ids)):.1f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / f"metrics_rank{args.rank}.csv", index=False)
    report = dict(run=str(args.run), split=args.split, checkpoint=args.checkpoint, checkpoint_step=ck["step"],
                  solver=args.solver, steps=args.steps, guidance=args.guidance, mean_samples=args.mean_samples,
                  refiner_ms_per_triplet=inference_seconds/len(indices)*1000, batch_size=4,
                  summary=summarize(df), seconds=time.monotonic()-start)
    (args.out / f"summary_rank{args.rank}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
