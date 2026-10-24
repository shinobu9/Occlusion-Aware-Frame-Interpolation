"""Aggregate primary metrics and paired bootstrap intervals without selecting on test."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.metrics import summarize


def paired_ci(d, rng, repeats=2000):
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(repeats)])
    return float(d.mean()), *map(float, np.quantile(boots, [.025, .975]))


def main():
    out = Path("artifacts")
    paths = sorted(out.glob("masks_test_rank*.csv"))
    assert len(paths) == 5, paths
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert not df.duplicated(["key", "method"]).any()
    assert (df.groupby("method").size() == 3782).all()
    summary = pd.DataFrame(summarize(df))
    df.to_csv(out / "mask_metrics_per_triplet.csv", index=False)
    summary.to_csv(out / "mask_summary.csv", index=False)
    rng = np.random.default_rng(42)
    rows = []
    for baseline in ["fixed"] + [m for m in df.method.unique() if m.startswith("heuristic_")]:
        for method in ("mask_s42", "mask_s43", "mask_s44"):
            for metric in ("psnr", "ssim", "mae"):
                pivot = df.pivot(index="key", columns="method", values=metric)
                d = (pivot[method]-pivot[baseline]).to_numpy()
                point, lo, hi = paired_ci(d, rng)
                better = lo > 0 if metric != "mae" else hi < 0
                worse = hi < 0 if metric != "mae" else lo > 0
                rows.append(dict(method=method, baseline=baseline, metric=metric, delta=point,
                                 ci_lo=lo, ci_hi=hi, verdict="better" if better else "worse" if worse else "inconclusive"))
    pd.DataFrame(rows).to_csv(out / "mask_paired_ci.csv", index=False)
    regional = []
    rrng = np.random.default_rng(420)
    for baseline in ["fixed"] + [x for x in df.method.unique() if x.startswith("heuristic_")]:
        for region in ("dis", "occ", "boundary"):
            p = df.pivot(index="key", columns="method", values=f"sse_{region}")
            a, b = p[baseline].to_numpy(), p.mask_s42.to_numpy()
            samples = []
            for _ in range(2000):
                ids = rrng.integers(0, len(a), len(a))
                samples.append(10*np.log10(a[ids].sum()/b[ids].sum()))
            lo, hi = np.quantile(samples, [.025, .975])
            regional.append(dict(baseline=baseline, region=region, delta=10*np.log10(a.sum()/b.sum()),
                                 ci_lo=lo, ci_hi=hi))
    pd.DataFrame(regional).to_csv(out / "mask_regional_ci.csv", index=False)
    fixed = df[df.method == "fixed"].set_index("key")
    learned = df[df.method == "mask_s42"].set_index("key")
    per = pd.DataFrame(dict(dis_fraction=fixed.n_dis/fixed.n_full,
                            delta_psnr=learned.psnr-fixed.psnr,
                            delta_mae=learned.mae-fixed.mae))
    per["dis_quartile"] = pd.qcut(per.dis_fraction, 4, labels=False, duplicates="drop")
    groups = []
    for q, d in per.groupby("dis_quartile"):
        point, lo, hi = paired_ci(d.delta_psnr.to_numpy(), rrng)
        groups.append(dict(quartile=int(q)+1, n=len(d), dis_fraction_min=d.dis_fraction.min(),
                           dis_fraction_max=d.dis_fraction.max(), delta_psnr=point,
                           ci_lo=lo, ci_hi=hi, delta_mae=d.delta_mae.mean()))
    pd.DataFrame(groups).to_csv(out / "mask_disagreement_subgroups.csv", index=False)
    print(summary.to_string(index=False))
    print("Primary mask seed fixed at 42; seeds 43/44 are robustness checks, not test-selected models.")


if __name__ == "__main__":
    main()
