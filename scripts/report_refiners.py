"""Consolidate test metrics and validation-only ablations; retain paired uncertainty."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.metrics import summarize
from scripts.report_masks import paired_ci


def main():
    root = Path("artifacts")
    validation = []
    for p in sorted(root.glob("val_*/summary_rank0.json")):
        r = json.loads(p.read_text())
        name = Path(r["run"]).name
        row = next(x for x in r["summary"] if x["method"] == name)
        validation.append(dict(experiment=p.parent.name, checkpoint_step=r["checkpoint_step"],
                               solver=r["solver"], steps=r["steps"], guidance=r["guidance"],
                               refiner_ms_per_triplet=r["refiner_ms_per_triplet"], **row))
    pd.DataFrame(validation).to_csv(root / "refiner_validation_summary.csv", index=False)
    summaries, cis = [], []
    rng = np.random.default_rng(42)
    for experiment in ("test_l1", "test_fm", "test_fm_mean4"):
        paths = sorted((root / experiment).glob("metrics_rank*.csv"))
        assert paths, (experiment, 'Run evaluate_refiner.py first')
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        assert not df.duplicated(["key", "method"]).any()
        assert (df.groupby("method").size() == 3782).all()
        df.to_csv(root / experiment / "metrics.csv", index=False)
        conf = json.loads((root / experiment / "summary_rank0.json").read_text())
        main_name = Path(conf["run"]).name
        for row in summarize(df):
            if row["method"] == "learned" and experiment != "test_l1":
                continue
            row.update(experiment=experiment, trained_model=main_name,
                       solver=conf["solver"], steps=conf["steps"], guidance=conf["guidance"], mean_samples=conf["mean_samples"])
            row["method"] = "learned_cached_reference" if row["method"] == "learned" else experiment.removeprefix("test_") + ("_dis" if row["method"].endswith("_dis") else "")
            summaries.append(row)
        for metric in ("psnr", "ssim", "mae", "lpips", "lpips_window"):
            pivot = df.pivot(index="key", columns="method", values=metric)
            d = (pivot[main_name]-pivot["learned"]).to_numpy()
            point, lo, hi = paired_ci(d, rng)
            lower = metric in ("mae", "lpips", "lpips_window")
            better, worse = (hi < 0, lo > 0) if lower else (lo > 0, hi < 0)
            cis.append(dict(experiment=experiment, metric=metric, delta=point, ci_lo=lo, ci_hi=hi,
                            verdict="better" if better else "worse" if worse else "inconclusive"))
    pd.DataFrame(summaries).to_csv(root / "refiner_test_summary.csv", index=False)
    pd.DataFrame(cis).to_csv(root / "refiner_paired_ci.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
