"""Synchronized batch-1 latency; raw repetitions are preserved."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.pipeline import Interpolator
from src.core import make_warps, make_features


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    root = Path(__file__).resolve().parents[1]
    strength = json.loads((root / "artifacts/heuristic_selection.json").read_text())["strength"]
    interp = Interpolator(root / "runs/mask_s42/best.pt", "cuda", strength)
    frames = torch.from_numpy(np.load(root / "artifacts/demo_triplets.npz")["frames"][0]).cuda().float()/255
    a, b = frames[0:1], frames[2:3]
    # Time the actual deployed learned path, excluding optional baseline visualizations.
    @torch.inference_mode()
    def shared():
        aa, bb = interp.preprocess(a, b)
        f01, f10 = interp.flow(aa, bb)[-1].half().float(), interp.flow(bb, aa)[-1].half().float()
        wd = make_warps(a, b, f01, f10)
        return make_features(wd).half().float()
    @torch.inference_mode()
    def learned(f):
        m = interp.mask(f)
        return m*f[:, :3]+(1-m)*f[:, 3:6]
    def timing(fn):
        for _ in range(5): fn()
        out = []
        for _ in range(30):
            torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize()
            out.append((time.perf_counter()-t)*1000)
        return out
    f = shared()
    times = dict(flow_warp_features=timing(shared), mask_and_blend=timing(lambda: learned(f)),
                 full_pipeline=timing(lambda: learned(shared())))
    report = dict(gpu=torch.cuda.get_device_name(), resolution="256x448", batch_size=1,
                  repetitions=30, warmup=5, torch=torch.__version__,
                  precision="fp32 with fp16 flow/features round-trip; TF32 convolution allowed",
                  **{f"{k}_median_ms": float(np.median(v)) for k, v in times.items()})
    (root / "artifacts/runtime.json").write_text(json.dumps(report, indent=2))
    (root / "artifacts/runtime_raw.json").write_text(json.dumps(times, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
