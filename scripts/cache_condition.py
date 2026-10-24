"""Freeze seed-42 mask predictions for the matched residual-refiner experiments."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.core import MaskUNet, blend
from src.data import CachedData, wd_from_features
from src.refiner import refiner_cond
from src.runtime import setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--mask", type=Path, default=Path("runs/mask_s42/best.pt"))
    args = ap.parse_args()
    setup(42)
    model = MaskUNet().cuda().eval()
    model.load_state_dict(torch.load(args.mask, map_location="cuda", weights_only=False)["model"])
    with torch.inference_mode():
        for s in ("train", "val", "test"):
            data = CachedData(args.data, s)
            out = np.lib.format.open_memmap(args.data / f"{s}_condition.npy", mode="w+", dtype=np.float16,
                                            shape=(len(data), 15, 256, 448))
            for i in range(0, len(data), 8):
                f, _, reg = data.batch(np.arange(i, min(i+8, len(data))))
                wd = wd_from_features(f)
                m = model(f)
                out[i:i+len(f)] = refiner_cond(wd, m, blend(wd, m), reg[:, :1]).half().cpu().numpy()
                if i % 400 == 0:
                    print(f"condition {s}: {i+len(f)}/{len(data)}", flush=True)
            out.flush()



if __name__ == "__main__":
    main()
