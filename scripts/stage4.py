"""Freeze validation choices, then run the complete test for prespecified comparisons."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage1 import ROOT, GPUS
from scripts.stage3 import run_queue


def main():
    os.chdir(ROOT)
    while not all((ROOT / f"runs/{name}.json").exists() for name in ("stage3_DONE", "stage3_combined_DONE")):
        print("waiting for validation diagnostics", flush=True)
        time.sleep(30)
    candidates = []
    for path in sorted((ROOT / "artifacts").glob("val_*/summary_rank0.json")):
        report = json.loads(path.read_text())
        name = Path(report["run"]).name
        if name == "l1":
            continue
        row = next(x for x in report["summary"] if x["method"] == name)
        candidates.append((row["pooled_psnr_dis"], report, str(path.relative_to(ROOT))))
    _, winner, source = max(candidates, key=lambda x: x[0])
    selection = dict(criterion="maximum pooled PSNR dis on fixed 64 validation triplets", source=source, config=winner,
                     note="No test metrics used for refiner/sampler/guidance selection")
    (ROOT / "artifacts/final_selection.json").write_text(json.dumps(selection, indent=2))
    experiments = [
        ("test_l1", dict(run="runs/l1", solver="euler", steps=20, guidance=1., mean_samples=1)),
        ("test_fm", dict(run="runs/fm", solver="euler", steps=20, guidance=1., mean_samples=1)),
        ("test_fm_mean4", dict(run="runs/fm", solver="euler", steps=20, guidance=1., mean_samples=4)),
        ("test_selected_fm", {k: winner[k] for k in ("run", "solver", "steps", "guidance", "mean_samples")}),
    ]
    tasks = []
    for tag, conf in experiments:
        for rank in range(len(GPUS)):
            cmd = ["scripts/evaluate_refiner.py", "--split", "test", "--out", f"artifacts/{tag}",
                   "--rank", str(rank), "--world", str(len(GPUS))]
            for key, value in conf.items():
                cmd += ["--"+key.replace("_", "-"), str(value)]
            tasks.append((f"{tag}_{rank}", cmd))
    # GPU 5 is released by the combined 30k experiment before this stage starts.
    run_queue(tasks, gpus=[1, 3, 4, 5, 6, 7])
    (ROOT / "runs/stage4_DONE.json").write_text(json.dumps(dict(completed=time.time())))


if __name__ == "__main__":
    main()
