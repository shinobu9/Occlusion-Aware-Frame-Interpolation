"""Explicitly test the colleague's combined EMA + logit-normal + 30k proposal."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage1 import ROOT
from scripts.stage3 import run_queue, eval_command

os.chdir(ROOT)
(ROOT / "artifacts/combined_30k_plan.json").write_text(json.dumps(dict(
    reason="Explicit control for the proposed EMA + logit-normal time + longer-training combination",
    settings=dict(steps=30000, ema=.999, time_density="logit_normal", cfg_dropout=0.),
    selection="validation only, same fixed 64 triplets", planned_before_test=True), indent=2))
name = "fm_ema_lognormal_30k"
run_queue([(name, ["scripts/train_refiner.py", "--run", f"runs/{name}", "--steps", "30000",
                   "--ema", ".999", "--time-density", "logit_normal"])], gpus=[5])
run_queue([("val_"+name, eval_command(name, "val_"+name))], gpus=[5])
(ROOT / "runs/stage3_combined_DONE.json").write_text(json.dumps(dict(completed=time.time())))
