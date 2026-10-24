"""Finite, prespecified refiner ablations. Uses validation only for model selection."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage1 import ROOT, GPUS, check_free, launch, finish
from scripts.stage3 import run_queue


def main():
    os.chdir(ROOT)
    while not (ROOT / "runs/stage1_DONE.json").exists():
        print("waiting for mask stage", flush=True)
        time.sleep(30)
    if not (ROOT / "data/condition_report.json").exists():
        run_queue([("condition", ["scripts/cache_condition.py"])])
    # Download perceptual weights once before parallel evaluation.
    import lpips
    lpips.LPIPS(net="alex", verbose=False)
    run_queue([(f"mask_test_{rank}", ["scripts/evaluate_masks.py", "--split", "test",
                  "--models", "runs/mask_s42", "runs/mask_s43", "runs/mask_s44", "--rank", str(rank), "--world", str(len(GPUS))]) for rank in range(len(GPUS))
               if not (ROOT / f"artifacts/masks_test_rank{rank}.csv").exists()])
    experiments = [
        ("l1", ["--mode", "l1"]),
        ("fm", []),
        ("fm_ema", ["--ema", ".999"]),
        ("fm_ema_lognormal", ["--ema", ".999", "--time-density", "logit_normal"]),
        ("fm_ema_lognormal_cfg", ["--ema", ".999", "--time-density", "logit_normal", "--cfg-dropout", ".1"]),
    ]
    (ROOT / "artifacts/experiment_plan.json").write_text(json.dumps(dict(
        primary="fixed vs validation-selected heuristic vs learned mask, three mask seeds",
        refiners=experiments, training_steps=12000, seed=42,
        validation="first 64 of fixed 150 validation triplets, fixed evaluation noise",
        sampler_sweep="Euler/Heun x 20/50/100 on validation only",
        longer_training="30k baseline FM and 30k strongest 12k FM recipe selected by validation",
        test="all 3782, no selection on test"), indent=2))
    run_queue([(name, ["scripts/train_refiner.py", "--run", f"runs/{name}", *options]) for name, options in experiments])
    (ROOT / "runs/stage2_DONE.json").write_text(json.dumps(dict(completed=time.time())))


if __name__ == "__main__":
    main()
