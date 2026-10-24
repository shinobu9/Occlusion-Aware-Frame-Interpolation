# Occlusion-Aware Frame Interpolation

**Lisa Kamenskaya · Pavel Vokhmyanin · Avdey Furmanyuk**

Given two endpoint frames, frozen RAFT-small estimates motion and forward-warps both frames to the midpoint. A 95,185-parameter CNN learns which warped frame to trust at each pixel. We compare it with fixed 50/50 and flow-consistency blending. This is a controlled study of established ideas, not a new algorithm or DLSS implementation.

## Start here

- [Executed notebook](occlusion_aware_frame_interpolation_final.ipynb): explanation, formulas, results and two-frame inference; training is disabled by default.
- Six-slide presentation: [PPTX with embedded video on slide 5](presentation/occlusion_aware_frame_interpolation.pptx) · [PDF](presentation/occlusion_aware_frame_interpolation.pdf).
- [Demo video](demo/interpolation_demo.mp4): **left = fixed baseline, center = our mask, right = ground truth**.
- [Results and experiment guide](results/README.md): what every experiment and CSV means, how the numbers were obtained, and limitations.

## Repository contents

| Path | Purpose |
|---|---|
| `src/` | Geometry, models, inference, data loader and metric definitions |
| `scripts/` | Data preparation, training, evaluation and result aggregation only |
| `checkpoints/` | Three inference models: primary mask, L1 refiner, baseline FM; each has its training config |
| `results/` | **Three final CSV files**, their explanation, and one study metadata JSON |
| `data/*_manifest.json` | Sequence ID lists for the published train/val/test split |
| `demo/`, `figures/` | Small held-out demo bundle, video and three result figures |
| `presentation/` | Final PDF and PPTX only |

Full datasets, caches, per-GPU CSV shards, training histories and optimizer states are **not tracked**. They are generated under ignored `data/`, `artifacts/` and `runs/` paths. The original complete experiment archive remains on Genesis at `/home/kamenskaya-el/hw/frame_interpolation`.

## Install and run inference

Tested with Python 3.12, PyTorch 2.14.1 + torchvision 0.29.1 (CUDA 12.6) on H100. Install a compatible torch/torchvision pair for your hardware, then the remaining requirements.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Open the notebook from the repository root. It uses saved tables and performs actual inference using only the two endpoint images. RAFT weights download on first use. Keep the notebook with the rest of the repository.

```python
from src.pipeline import Interpolator
model = Interpolator('checkpoints/mask_s42/best.pt')
# earlier/later: float RGB tensors [3,H,W] in [0,1]
midpoint = model(earlier, later)['learned']
```

## Reproduce the primary experiment

Use a fresh data/cache directory if changing the split or preprocessing. Full data and caches require roughly 55 GB. Set `CUDA_VISIBLE_DEVICES` to a free GPU. The scripts below default to one GPU; no host-specific GPU scheduler is needed. Training is resumable, and directories with `DONE.json` are skipped; use new run directories for a fresh repeat.

```bash
python scripts/prepare_data.py
python scripts/cache_flows.py --init
python scripts/cache_flows.py --rank 0 --world 1
python scripts/evaluate_masks.py --split val
for seed in 42 43 44; do
  python scripts/train_mask.py --run runs/mask_s${seed} --seed ${seed}
done
python scripts/evaluate_masks.py --split test --models runs/mask_s42 runs/mask_s43 runs/mask_s44
python scripts/report_masks.py
python scripts/evaluate_interior.py
python scripts/benchmark.py
```

Dataset sources: [training/validation mirror](https://huggingface.co/datasets/bijinc/vimeo-90k-medium) and [test mirror](https://huggingface.co/datasets/paschalc24/vimeo-90k-test-full). `prepare_data.py` downloads their current files and selects the sequence IDs listed in `data/*_manifest.json`. These lists preserve the published split. Train and validation use different directory-prefix groups; these are not verified original-video IDs, so similar scenes may remain.

## Reproduce the refiner comparison

The mask stays frozen. L1 and FM share the same 3.31M-parameter architecture, crop stream and conditioning. Each best checkpoint is selected by validation disagreement-region PSNR.

```bash
python scripts/cache_condition.py --mask runs/mask_s42/best.pt
python scripts/train_refiner.py --run runs/l1 --mode l1
python scripts/train_refiner.py --run runs/fm
python scripts/evaluate_refiner.py --run runs/l1 --split test --out artifacts/test_l1
python scripts/evaluate_refiner.py --run runs/fm --split test --out artifacts/test_fm
python scripts/evaluate_refiner.py --run runs/fm --split test --mean-samples 4 --out artifacts/test_fm_mean4
```

To re-evaluate the supplied weights instead of retraining, replace the relevant `runs/...` arguments with `checkpoints/...`; use `--mask checkpoints/mask_s42/best.pt` for conditioning. To recreate all validation ablations, run the commands in [the experiment guide](results/README.md). Then aggregate and export:

```bash
python scripts/report_refiners.py
python scripts/export_results.py
```

Evaluation writes per-triplet measurements to ignored `artifacts/`. The reports aggregate them, compute paired bootstrap intervals and export the three public tables in `results/`. Aggregation supports either one evaluation shard or several disjoint shards. Dataset IDs, configs and seeds are fixed; CUDA accumulation is not guaranteed bitwise deterministic.

The original team notebooks supplied the forward-splat, MaskUNet and refiner starting code. The rerun corrects data overlap, heuristic-strength propagation, invalid flow handling, SSIM reporting, checkpoint selection and confidence-interval interpretation. It is an adapted experiment, not an exact replay of their original outputs.
