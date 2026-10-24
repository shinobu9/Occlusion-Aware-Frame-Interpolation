# Results and experiment guide

These are completed reruns, not the saved outputs of the exploratory notebooks. Training uses 1,200 Vimeo-90K triplets, validation 150, and testing the full 3,782-triplet public test mirror. Refiner diagnostics use a fixed subset of 64 validation triplets. The middle frame is ground truth; only the two endpoints enter inference.

## The three CSV files

| File | What one row means | How it is produced |
|---|---|---|
| [test_metrics.csv](test_metrics.csv) | One method evaluated on all 3,782 test triplets. `family` separates the primary mask study from residual refiners; `description` explains the method. | `evaluate_masks.py` / `evaluate_refiner.py` → `report_masks.py` / `report_refiners.py` → `export_results.py` |
| [validation_ablations.csv](validation_ablations.csv) | One training/sampling configuration on 64 fixed validation triplets. `description` states what changed. These rows are **not test results**. | `evaluate_refiner.py --split val --limit 64` → `report_refiners.py` → `export_results.py` |
| [paired_comparisons.csv](paired_comparisons.csv) | Method minus baseline, with a paired 95% bootstrap interval. `scope` is full frame, a diagnostic region, or the interior after removing a 16px border. | The report scripts and `evaluate_interior.py` → `export_results.py` |

[study.json](study.json) contains dataset links and split sizes, the selected heuristic strength, runtime measurements, subgroup analysis and demo reference scores. The notebook reads these saved results; it does not silently train new models.

**Metrics:** PSNR/SSIM/MAE are averages over images; MAE/MSE use RGB in [0,1]. SSIM uses a 7×7 uniform window and sample covariance. Regional PSNR pools squared error and pixel counts. `dis` is warp disagreement >0.04, `occ` is a flow-consistency/coverage proxy, and `boundary` is flow variation >1 pixel. These are not ground-truth visibility annotations. LPIPS is measured for the refiner study; empty mask-family LPIPS cells mean it was not measured in that evaluation. `lpips_window` uses the 128×128 window with greatest warp disagreement.

For paired differences, positive PSNR/SSIM is better; negative MAE/LPIPS is better. An interval crossing zero is inconclusive. Intervals resample triplets, not unknown original-video clusters; they are not corrected for all exploratory comparisons.

## What was tested

| Method / experiment | Purpose |
|---|---|
| `fixed` | Main baseline 1: equal blending of the two shared warps |
| `heuristic_0.5` | Main baseline 2: flow-consistency blending; strength 0.5 selected on validation |
| `mask_s42` | Our primary learned mask; seed fixed in advance, 8k training steps |
| `mask_s43`, `mask_s44` | Same training, two other seeds; robustness checks |
| `unwarped`, `old_half_flow` | Auxiliary geometry controls: no motion compensation / approximate backward half-flow |
| `oracle` | Ground-truth-informed convex blend; squared-error ceiling, not an inference method |
| `learned_cached_reference` | Same seed-42 mask rounded into the shared float16 refiner cache; tiny metric differences from the primary row are expected |
| `l1` | Deterministic residual refiner; 12k training steps |
| `fm` | Conditional flow-matching residual refiner; 12k steps, best validation checkpoint at step 2k, Euler 20 |
| `fm_mean4` | Same FM checkpoint; average four independently seeded samples |

The repeated “validation-selected FM” test was identical to `fm`; its duplicate rows and files have been removed. Per-GPU shards and regional-only refiner variants are intermediate diagnostics, not additional headline methods.

## Validation ablations: exact commands

Run after preparing the shared conditioning and training the baseline FM/L1 above. Refiner seed is 42. EMA decay is 0.999; logit-normal means sigmoid(N(0,1)); condition dropout is 0.1. Each 30k run starts from scratch with a 30k cosine schedule.

```bash
python scripts/train_refiner.py --run runs/fm_ema --ema .999
python scripts/train_refiner.py --run runs/fm_ema_lognormal --ema .999 --time-density logit_normal
python scripts/train_refiner.py --run runs/fm_ema_lognormal_cfg --ema .999 --time-density logit_normal --cfg-dropout .1
python scripts/train_refiner.py --run runs/fm_30k --steps 30000
python scripts/train_refiner.py --run runs/fm_ema_lognormal_30k --steps 30000 --ema .999 --time-density logit_normal
for solver in euler heun; do
  for steps in 20 50 100; do
    python scripts/evaluate_refiner.py --run runs/fm --split val --limit 64 --solver ${solver} --steps ${steps} --out artifacts/val_fm_${solver}${steps}
  done
done
for name in l1 fm_ema fm_ema_lognormal fm_ema_lognormal_cfg fm_30k fm_ema_lognormal_30k; do
  python scripts/evaluate_refiner.py --run runs/${name} --split val --limit 64 --out artifacts/val_${name}
done
for w in 2 3; do
  python scripts/evaluate_refiner.py --run runs/fm_ema_lognormal_cfg --split val --limit 64 --guidance ${w} --out artifacts/val_cfg_w${w}
done
```

Sampler variants share fixed per-triplet noise. Checkpoint monitoring uses a separate fixed noise stream. CFG >1 is only valid for a checkpoint trained with condition dropout. Checkpoints and hyperparameters use validation, never test, for selection.

## Interpretation

The primary learned mask improves mean PSNR by **+0.334 dB** over fixed blending, with 95% CI **[+0.314, +0.356]**; all three seeds improve. After removing a 16px border, the gain remains **+0.234 dB**. This border diagnostic was added after inspecting examples and was not used to retrain or select models.

L1 reaches **33.405 dB**. Single-sample FM reaches **32.031 dB**, below the mask's **32.917 dB**. FM averaging reaches **32.704 dB** and improves LPIPS relative to the mask, but L1 is better on both metrics. More ODE steps, Heun, stronger CFG and the tested longer-training recipes did not improve the baseline FM's validation disagreement PSNR. This is a result for this small configuration, not a general claim against flow matching.

The improvement is modest; difficult motion and missing detail still cause failures. The training subset is small, the refiner study has one seed, and source-video correlations are unknown. Learned visibility-aware blending is established; the contribution here is the controlled implementation and analysis.

## Related work

- [Super SloMo — Jiang et al., CVPR 2018](https://research.nvidia.com/publication/2018-06_super-slomo-high-quality-estimation-multiple-intermediate-frames-video): learned visibility and flow refinement.
- [Softmax Splatting — Niklaus & Liu, CVPR 2020](https://arxiv.org/abs/2003.05534): weighted forward warping and synthesis.
- [RIFE — Huang et al., ECCV 2022](https://arxiv.org/abs/2011.06294): learned intermediate flow.
- [RAFT — Teed & Deng, 2020](https://arxiv.org/abs/2003.12039): our frozen motion estimator.
- [Flow Matching — Lipman et al., 2022](https://arxiv.org/abs/2210.02747), [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598), [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206): the generative extension and ablation ideas.
