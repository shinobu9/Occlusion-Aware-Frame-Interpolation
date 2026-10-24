"""Write evidence-backed conclusions and rerun instructions from completed artifacts."""
import importlib.metadata as meta
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def main():
    shutil.copy2(ROOT / "data/data_report.json", ART / "data_report.json")
    shutil.copy2(ROOT / "data/condition_report.json", ART / "condition_report.json")
    m = pd.read_csv(ART / "mask_summary.csv").set_index("method")
    r = pd.read_csv(ART / "refiner_test_summary.csv").set_index("method")
    ci = pd.read_csv(ART / "mask_paired_ci.csv")
    selected = json.loads((ART / "final_selection.json").read_text())
    groups = pd.read_csv(ART / "mask_disagreement_subgroups.csv")
    diagnostics = pd.read_csv(ART / "refiner_validation_summary.csv").set_index("experiment")
    interior_ci = pd.read_csv(ART / "mask_interior_ci.csv")
    interior = interior_ci[(interior_ci.metric == "psnr") & (interior_ci.baseline == "fixed")].iloc[0]
    delta = m.loc["mask_s42", "psnr"]-m.loc["fixed", "psnr"]
    row = ci[(ci.method == "mask_s42") & (ci.baseline == "fixed") & (ci.metric == "psnr")].iloc[0]
    heur = next(k for k in m.index if k.startswith("heuristic_"))
    text = f"""The primary seed-42 learned mask improves mean test PSNR by {delta:+.3f} dB over fixed blending (paired triplet-bootstrap 95% CI [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]) and by {m.loc['mask_s42','psnr']-m.loc[heur,'psnr']:+.3f} dB over the validation-selected heuristic.
SSIM changes from {m.loc['fixed','ssim']:.5f} to {m.loc['mask_s42','ssim']:.5f}; MAE changes from {m.loc['fixed','mae']:.6f} to {m.loc['mask_s42','mae']:.6f}.
All three mask seeds improve mean test PSNR over fixed blending. Regional results support benefits in the chosen disagreement/occlusion-related proxies, but these are not ground-truth occlusion annotations.
Gains are positive in all four warp-disagreement quartiles, but they are not monotonic: the lowest-disagreement quartile gains {groups.iloc[0].delta_psnr:+.3f} dB and the highest gains {groups.iloc[-1].delta_psnr:+.3f} dB. Larger absolute MAE reductions in difficult examples do not imply larger dB gains. This is consistent with limits of blending under imperfect warping, but does not establish the cause of each failure.
Post-hoc border diagnostic: after removing a 16-pixel border on all sides, the learned-mask gain over fixed blending is {interior.delta:+.3f} dB (paired 95% CI [{interior.ci_lo:+.3f}, {interior.ci_hi:+.3f}]). This check was added after inspecting qualitative examples; it was not used to select or retrain any model.
The L1 refiner reaches {r.loc['l1','psnr']:.3f} dB and LPIPS {r.loc['l1','lpips']:.4f}. Baseline FM reaches {r.loc['fm','psnr']:.3f} dB and LPIPS {r.loc['fm','lpips']:.4f}. Validation-selected FM reaches {r.loc['selected_fm','psnr']:.3f} dB and LPIPS {r.loc['selected_fm','lpips']:.4f}.
Four-sample FM averaging reaches {r.loc['fm_mean4','psnr']:.3f} dB and LPIPS {r.loc['fm_mean4','lpips']:.4f}: its LPIPS improves over the mask-only reference, while PSNR remains lower. Thus FM averaging shows a metric trade-off, rather than failure on every metric. L1 still performs better on both metrics and uses a single refiner call.
Validation selected {selected['config']['run']} at checkpoint step {selected['config']['checkpoint_step']}, with {selected['config']['solver']} {selected['config']['steps']} and guidance {selected['config']['guidance']:g}. If this matches the baseline configuration, the selected-FM row is a repeated evaluation of that same model, not a separate improved method.
The strongest tested refiner by mean test PSNR is {r.loc[['l1','fm','fm_mean4','selected_fm'],'psnr'].idxmax()}; this is a descriptive comparison, not a test-driven model-selection rule.
Validation sampler diagnostics: Euler 20 gives pooled disagreement PSNR {diagnostics.loc['val_fm_euler20','pooled_psnr_dis']:.3f} dB; Euler 100 gives {diagnostics.loc['val_fm_euler100','pooled_psnr_dis']:.3f}; Heun 100 gives {diagnostics.loc['val_fm_heun100','pooled_psnr_dis']:.3f}. More integration steps do not guarantee a better reconstruction with an imperfect learned velocity field.
On the condition-dropout checkpoint, guidance weights 1 / 2 / 3 give validation disagreement PSNR {diagnostics.loc['val_fm_ema_lognormal_cfg','pooled_psnr_dis']:.3f} / {diagnostics.loc['val_cfg_w2','pooled_psnr_dis']:.3f} / {diagnostics.loc['val_cfg_w3','pooled_psnr_dis']:.3f} dB. This sweep does not support stronger CFG for this configuration.
Longer-training controls: baseline FM 30k gives validation disagreement PSNR {diagnostics.loc['val_fm_30k','pooled_psnr_dis']:.3f} dB; EMA + logit-normal + 30k gives {diagnostics.loc['val_fm_ema_lognormal_30k','pooled_psnr_dis']:.3f} dB. These are separate from-scratch learning-rate schedules, evaluated through their best validation checkpoints.
The study establishes a result for this limited configuration. Learned visibility-aware blending is an established idea; our contribution is implementation and controlled analysis with frozen RAFT-small. The study is neither a new interpolation algorithm nor an exact reproduction of a prior paper.
Limitations: 1,200 training triplets, three mask seeds but one refiner seed, unknown source-video correlations, proxy visibility regions, and imperfect flow. The oracle uses ground truth and only bounds squared error for convex blending. Statistical intervals resample triplets.
"""
    (ART / "conclusions.txt").write_text(text)
    packages = ["torch", "torchvision", "numpy", "pandas", "pyarrow", "scikit-image", "lpips", "matplotlib", "Pillow", "requests", "tqdm", "nbformat", "nbclient", "ipykernel", "python-pptx", "imageio", "imageio-ffmpeg"]
    versions = {p: meta.version(p) for p in packages}
    (ROOT / "requirements.txt").write_text("\n".join(f"{p}=={v.split('+')[0]}" for p, v in versions.items())+"\n")
    (ART / "environment.json").write_text(json.dumps(versions, indent=2))
    rows = "\n".join(f"| {name} | {m.loc[k,'psnr']:.3f} | {m.loc[k,'ssim']:.5f} | {m.loc[k,'mae']:.6f} |" for k, name in [("fixed", "Fixed 50/50"), (heur, "Heuristic"), ("mask_s42", "Learned mask (seed 42)")])
    readme = f"""# Occlusion-Aware Frame Interpolation

**Lisa Kamenskaya · Pavel Vokhmyanin · Avdey Furmanyuk**

A small research prototype: frozen RAFT-small estimates bidirectional motion, both frames are forward-warped to the midpoint, and a 95,185-parameter CNN learns a spatial blending mask. This project studies established ideas; it does not claim a novel algorithm, reproduce a prior paper exactly, or implement DLSS.

## Deliverables

- `occlusion_aware_frame_interpolation_final.ipynb`: executed English notebook with analysis and two-frame inference.
- `presentation/occlusion_aware_frame_interpolation.pdf`: 10-slide defense presentation.
- `presentation/occlusion_aware_frame_interpolation.pptx`: matching slides (rendered slide images; rebuild from `scripts/build_presentation.py` to edit content).
- `artifacts/interpolation_demo.mp4`: slowed three-frame loops, fixed vs learned vs ground truth.
- `runs/*/best.pt`: selected trained weights; `config.json` and `history.jsonl`: exact settings and histories.
- `artifacts/*csv`: per-triplet metrics, aggregate results, paired confidence intervals and validation ablations.
- `artifacts/notebook_audit.md`: audit of the supplied notebooks and rationale for corrections.

## Main result

All metrics below use the full 3,782-triplet test mirror. PSNR is the mean of per-image PSNR, SSIM is the mean RGB SSIM with a 7×7 uniform window, and MAE is on [0,1].

| Method | PSNR ↑ | SSIM ↑ | MAE ↓ |
|---|---:|---:|---:|
{rows}

The seed-42 gain over fixed blending is **{delta:+.3f} dB**, with paired triplet-bootstrap 95% CI **[{row.ci_lo:+.3f}, {row.ci_hi:+.3f}]**. See `artifacts/conclusions.txt` for the complete interpretation and limitations. The other two seeds are robustness checks; seed 42 was fixed in advance as the primary model.

## Setup and inference

Tested with Python 3.12 on H100. Use an isolated environment. Install a matching torch/torchvision pair for your hardware; the measured run used `{versions['torch']}` and `{versions['torchvision']}`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch=={versions['torch'].split('+')[0]} torchvision=={versions['torchvision'].split('+')[0]} --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Open the notebook from the repository root. By default it loads saved completed results and performs two-frame inference on the bundled demo triplets; it does not launch training. A GPU is recommended, but the mask pipeline also has a CPU inference path.

```python
from src.pipeline import Interpolator
model = Interpolator('runs/mask_s42/best.pt')
# earlier/later: float RGB tensors [3,H,W] in [0,1]
result = model(earlier, later)
midpoint = result['learned']
```

Ground truth is never passed to the predictor. The flow model weights are downloaded from torchvision on first use. The oracle is only used offline in evaluation.

## Reproduce the data and models

The full cache uses roughly 55 GB of disk. For a from-scratch rerun, create a new working directory containing `src/`, `scripts/` and `requirements.txt`; do not copy completed `runs/` directories or cache completion markers. GPU jobs should be assigned only to GPUs without existing compute processes. The following commands are single-GPU building blocks; replace device 0 with a genuinely free GPU.

```bash
python scripts/sanity.py
python scripts/prepare_data.py
python scripts/cache_flows.py --init
CUDA_VISIBLE_DEVICES=0 python scripts/cache_flows.py --rank 0 --world 1
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_masks.py --split val
CUDA_VISIBLE_DEVICES=0 python scripts/train_mask.py --run runs/mask_s42 --seed 42
CUDA_VISIBLE_DEVICES=0 python scripts/train_mask.py --run runs/mask_s43 --seed 43
CUDA_VISIBLE_DEVICES=0 python scripts/train_mask.py --run runs/mask_s44 --seed 44
CUDA_VISIBLE_DEVICES=0 python scripts/cache_condition.py
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py --run runs/l1 --mode l1
CUDA_VISIBLE_DEVICES=0 python scripts/train_refiner.py --run runs/fm
```

Optional FM ablations: `--ema .999`, then `--time-density logit_normal`, then `--cfg-dropout .1`. The longer controls use a separate run directory and `--steps 30000`; they start from scratch with a 30k cosine schedule, rather than resuming a completed 12k schedule. `scripts/evaluate_refiner.py` accepts `--solver euler/heun`, `--steps 20/50/100`, `--guidance 1/2/3`, and `--mean-samples 4`. CFG >1 is rejected for models not trained with condition dropout.

The Genesis orchestration scripts `stage1.py`–`stage4.py` contain the exact completed study plan and use five evaluation shards. `stage3_combined.py` additionally checks the explicitly proposed EMA + logit-normal + 30k combination, and stage 4 waits for it. They are machine-specific: inspect the candidate GPU lists before reusing them. Stages 2–4 dynamically skip occupied GPUs before launching each child. Their tasks run in tmux and write logs; failures stop dependent work. Checkpoints preserve optimizer, scheduler and RNG state. Completed trainings are not rerun automatically. Evaluation aggregation expects all five shards; individual evaluation scripts can also run with `--world 1` for a standalone CSV. Use a fresh data directory if changing the split or preprocessing settings.

## Data protocol and evaluation

Training: 1,200 triplets; validation: 150; full test: 3,782. Dataset revisions, source parquet hashes, selected sequence IDs and decoded-frame hashes are preserved in `data/*manifest.json` and `artifacts/data_report.json` on Genesis. The small local delivery contains manifests and a few held-out demo triplets, not the full dataset/cache.

- Training source: [bijinc/vimeo-90k-medium](https://huggingface.co/datasets/bijinc/vimeo-90k-medium), revision `17ba080aa556c0050d6c560115259e26c03291f9`.
- Test source: [paschalc24/vimeo-90k-test-full](https://huggingface.co/datasets/paschalc24/vimeo-90k-test-full), revision `7af0ffe8b63dabcadb4dee8cf8c543e269dffb5b`.
- All selected train/validation IDs and decoded RGB-frame hashes are checked against the full test. Directory-prefix groups separate train from validation; these are not verified original-video IDs. Similar scenes may remain.
- Flow is rounded to float16 once; warping uses float32, and shared features are cached in float16 for all primary methods. Refiner conditioning includes a float16 cached blend; refiner comparisons use that same cached reference.
- Standard per-image PSNR and pooled regional PSNR are reported separately. Region `dis` means warp disagreement >0.04; `occ` means flow-consistency/coverage proxy; `boundary` means flow-variation proxy. None is ground-truth visibility.
- Hyperparameters and checkpoints are chosen on validation. Refiner tuning uses the first 64 fixed validation triplets with fixed noise; this small tuning subset is a limitation.
- Checkpoint monitoring and sampler diagnostics use separate fixed noise streams. Sampler settings share per-triplet diagnostic noise, and the same scheme is used for final testing and figures.
- Bootstrap intervals resample paired triplets. They do not account for unknown source-video clusters or correct for all exploratory comparisons.

## Changes from the supplied notebooks

Original notebooks are preserved unchanged in `originals/`, with SHA-256 checksums. Their embedded exploratory metrics are not reused as final results.

We reuse the forward-splat pipeline, MaskUNet and RefinerUNet from `fps_generator_kaggle.ipynb`. We correct the data split, explicit heuristic-strength propagation, double-hole fallback, out-of-frame consistency checks, SSIM and CI interpretation. We use a shared immutable cache, best validation checkpoints, fixed evaluation noise and three mask seeds. Mask crops are sampled independently per example after full-frame warping. Refiner conditioning is cached from the primary mask, and temporal swapping is omitted for these cached directional conditions. These deliberate changes mean the new experiment is not an exact replay of the colleagues' original runs. The first notebook's approximate half-flow geometry is retained as the `old_half_flow` baseline, rather than silently conflated with the forward-splat comparison.

## Related work

- Jiang et al. **Super SloMo**, CVPR 2018: [paper/project](https://research.nvidia.com/publication/2018-06_super-slomo-high-quality-estimation-multiple-intermediate-frames-video).
- Niklaus & Liu. **Softmax Splatting for Video Frame Interpolation**, CVPR 2020: [paper](https://arxiv.org/abs/2003.05534).
- Huang et al. **RIFE**, ECCV 2022: [paper](https://arxiv.org/abs/2011.06294).
- Teed & Deng. **RAFT**, 2020: [paper](https://arxiv.org/abs/2003.12039).
- Lipman et al. **Flow Matching for Generative Modeling**, 2022: [paper](https://arxiv.org/abs/2210.02747).
- Ho & Salimans. **Classifier-Free Diffusion Guidance**, 2022: [paper](https://arxiv.org/abs/2207.12598).
- Esser et al. **Scaling Rectified Flow Transformers**, 2024: [paper](https://arxiv.org/abs/2403.03206).

Our contribution is a controlled empirical comparison in a restricted configuration, with transparent failure cases and computational cost. It is not novelty of learned visibility-aware blending.
"""
    (ROOT / "README.md").write_text(readme)
    (ROOT / ".gitignore").write_text(".venv/\n__pycache__/\n.ipynb_checkpoints/\ndata/*npy\ndata/downloads/\nruns/*/latest.pt\nlogs/\n*.partial\n")
    print("Wrote README, dependencies, environment and conclusions")


if __name__ == "__main__":
    main()
