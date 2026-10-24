"""Validation sampler sweep, CFG strengths, and longer-training controls."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scripts.stage1 import ROOT, GPUS, available_gpus, launch


def run_queue(tasks, gpus=GPUS):
    queue = list(tasks)
    active = {}
    while queue or active:
        for gpu in available_gpus(gpus) if queue else []:
            if gpu not in active and queue:
                name, command = queue.pop(0)
                active[gpu] = launch(gpu, name, command)
        for gpu, (name, proc, log) in list(active.items()):
            rc = proc.poll()
            if rc is not None:
                log.close()
                print(f"FINISH {name} rc={rc}", flush=True)
                del active[gpu]
                if rc:
                    # Let already-running independent jobs finish; do not launch further work.
                    for _, child, logfile in active.values():
                        child.wait()
                        logfile.close()
                    raise RuntimeError(f"{name} failed: {rc}")
        if active or queue:
            time.sleep(5)


def eval_command(run, out, **kwargs):
    cmd = ["scripts/evaluate_refiner.py", "--run", f"runs/{run}", "--out", f"artifacts/{out}", "--split", "val", "--limit", "64"]
    for k, v in kwargs.items():
        cmd += ["--"+k.replace("_", "-"), str(v)]
    return cmd


def main():
    os.chdir(ROOT)
    while not (ROOT / "runs/stage2_DONE.json").exists():
        print("waiting for 12k refiners", flush=True)
        time.sleep(30)
    names = ["fm", "fm_ema", "fm_ema_lognormal", "fm_ema_lognormal_cfg"]
    scores = {n: torch.load(ROOT / f"runs/{n}/best.pt", map_location="cpu", weights_only=False)["val"]["dis"] for n in names}
    winner = max(scores, key=scores.get)
    (ROOT / "artifacts/long_training_selection.json").write_text(json.dumps(dict(scores=scores, selected=winner,
        criterion="best validation pooled PSNR dis, 64 fixed triplets; no test access"), indent=2))
    tasks = []
    for name in dict.fromkeys(["fm", winner]):
        config = json.loads((ROOT / f"runs/{name}/config.json").read_text())
        cmd = ["scripts/train_refiner.py", "--run", f"runs/{name}_30k", "--steps", "30000",
               "--ema", str(config["ema"]), "--time-density", config["time_density"],
               "--cfg-dropout", str(config["cfg_dropout"])]
        tasks.append((name+"_30k", cmd))
    for solver in ("euler", "heun"):
        for steps in (20, 50, 100):
            tag = f"val_fm_{solver}{steps}"
            tasks.append((tag, eval_command("fm", tag, solver=solver, steps=steps)))
    for name in ("l1", "fm_ema", "fm_ema_lognormal", "fm_ema_lognormal_cfg"):
        tasks.append((f"val_{name}", eval_command(name, f"val_{name}")))
    for guidance in (2, 3):
        tag = f"val_cfg_w{guidance}"
        tasks.append((tag, eval_command("fm_ema_lognormal_cfg", tag, guidance=guidance)))
    run_queue(tasks)
    tasks = [(f"val_{name}_30k", eval_command(f"{name}_30k", f"val_{name}_30k")) for name in dict.fromkeys(["fm", winner])]
    run_queue(tasks)
    (ROOT / "runs/stage3_DONE.json").write_text(json.dumps(dict(completed=time.time(), winner=winner)))


if __name__ == "__main__":
    main()
