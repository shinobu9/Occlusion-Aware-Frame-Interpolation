"""Build a readable notebook; expensive training is an explicit opt-in."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
n = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
n.cells = [
md("""# Occlusion-Aware Frame Interpolation
**Lisa Kamenskaya · Pavel Vokhmyanin · Avdey Furmanyuk**

Can a lightweight learned blending mask improve intermediate-frame reconstruction, especially around motion boundaries and occlusion-related regions, compared with fixed or heuristic blending?

This is a controlled study of established ideas, not a new interpolation algorithm, an exact reproduction of a prior paper, or an implementation of DLSS. Two supplied exploratory notebooks provided the starting code. We corrected the data/evaluation protocol and reran the experiments from scratch.
"""),
md("""## Related work and scope
- [Super SloMo, CVPR 2018](https://research.nvidia.com/publication/2018-06_super-slomo-high-quality-estimation-multiple-intermediate-frames-video): learned visibility maps and refined intermediate flow.
- [Softmax Splatting, CVPR 2020](https://arxiv.org/abs/2003.05534): weighted forward warping and learned image synthesis.
- [RIFE, ECCV 2022](https://arxiv.org/abs/2011.06294): directly learned intermediate flow.
- [RAFT, 2020](https://arxiv.org/abs/2003.12039): the optical-flow architecture used by our frozen motion estimator.

Our narrower experiment freezes **RAFT-small** and shares the same warps across the three primary methods. Only the blending CNN is trained. We evaluate the quality/cost trade-off and the limits of this restricted model. The residual L1/FM refiners are a separate extension, not part of the original mask-only method.
"""),
md("""## Run this notebook
Open it from the repository root. Install the dependencies in `requirements.txt` with a compatible PyTorch/torchvision pair. The saved tables, figures, checkpoints and small demo bundle are enough for the default analysis/inference path. Full training data and caches are only needed for a rerun. Training is disabled by default; every displayed experimental number comes from a saved completed run.
"""),
code("""from pathlib import Path
import json, math, time, sys, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Image, Video
get_ipython().run_line_magic('matplotlib', 'inline')
ROOT = Path.cwd()
if not (ROOT / 'src').exists():
    raise RuntimeError('Open the notebook from the repository root.')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name()}')
RUN_TRAINING = False
"""),
md("""## Data and leakage checks
The inputs are `im1` and `im3`; `im2` is ground truth and is never used by the deployed predictor. We pin both Hugging Face dataset revisions and preserve sequence IDs. Training candidates are filtered against **all 3,782 test sequences** by ID and decoded RGB-frame SHA-256. Exact repeated frames are also excluded across the selected train/validation pool.

Train and validation use disjoint directory-prefix groups. These prefixes are **not verified original-video IDs**; this does not guarantee absence of similar scenes. The public mirror describes its source as a subset of the official Vimeo triplet split. We additionally verify exact non-overlap ourselves.
"""),
code("""data_report = json.loads((ROOT / 'artifacts/data_report.json').read_text())
display(pd.Series(data_report['counts'], name='triplets').to_frame())
print(f"Excluded candidates: {len(data_report['excluded'])}")
print(data_report['limitation'])
"""),
md(r"""## Shared pipeline
1. Estimate bidirectional endpoint flow with frozen `Raft_Small_Weights.C_T_V2`.
2. Forward-splat both images by half their endpoint flow. Resolve collisions using brightness-consistency weights; use the approximate backward half-flow warp only to fill holes.
3. Blend the two midpoint warps:

\[
\hat I_t=M\odot W_0+(1-M)\odot W_1.
\]

**Fixed:** $M=0.5$. **Heuristic:** visibility/coverage cues from forward-backward consistency, with strength selected on validation. **Learned:** an 18-channel input and a three-level U-Net predict a sigmoid mask. All three use the same cached warps. Maps are transported onto the midpoint grid before use. When both warps have holes, the heuristic uses symmetric fallback blending.
"""),
code("""from src.core import MaskUNet, make_warps, make_features, heuristic_mask, blend, oracle_alpha
from src.pipeline import Interpolator
from src.metrics import measure, ssim
model = MaskUNet()
print(f'Mask parameters: {sum(p.numel() for p in model.parameters()):,}')
display(model)
"""),
md("""## Training protocol
The mask uses weighted L1 reconstruction loss, emphasizing pixels where the warps disagree. We run 8,000 optimizer steps for seeds 42, 43 and 44, with batch size 8 and 192×192 crops **after full-frame warping**. We select each checkpoint by validation reconstruction in the disagreement region. Seed 42 is the primary model fixed in advance; other seeds assess robustness.

The last heuristic feature channel has fixed strength 1.0 for every model. Tuning the standalone heuristic baseline does not change the trained model's features. We cache flow/features in float16 and calculate warping, model predictions and metrics in float32. Random seeds are recorded; CUDA accumulation is not guaranteed bitwise deterministic.
"""),
code("""# Optional single-GPU retraining after preparing the full data/cache.
if RUN_TRAINING:
    subprocess.run([sys.executable, 'scripts/train_mask.py', '--run', 'runs/mask_rerun',
                    '--seed', '42'], check=True)
else:
    print('Using saved completed experiments. See README for the full rerun commands.')
"""),
md("""## Metrics and interpretation
We report the mean of per-image RGB PSNR, SSIM and MAE on [0,1]. SSIM uses a 7×7 uniform window and sample covariance, matching the default `skimage.metrics.structural_similarity` configuration. Regional PSNR is computed from pooled squared error and pixel counts, and must not be confused with mean per-image PSNR.

- `dis`: mean absolute disagreement of the warps exceeds 0.04.
- `occ`: proxy region from flow consistency or missing splat coverage.
- `boundary`: proxy from spatial flow variation exceeding 1 pixel.

These are method-independent diagnostic regions, **not ground-truth occlusion labels**. Paired 95% bootstrap intervals resample triplets; they do not account for unknown source-video correlations. The GT-informed oracle is only a bound on the best convex RGB blend's squared error, not a deployable method or a perceptual-quality bound.
"""),
code("""primary = pd.read_csv(ROOT / 'artifacts/mask_summary.csv')
display(primary[['method', 'n_triplets', 'psnr', 'ssim', 'mae',
                 'pooled_psnr_dis', 'pooled_psnr_occ', 'pooled_psnr_boundary']].round(5))
ci = pd.read_csv(ROOT / 'artifacts/mask_paired_ci.csv')
display(ci[ci.method == 'mask_s42'].round(6))
display(pd.read_csv(ROOT / 'artifacts/mask_regional_ci.csv').round(4))
display(pd.read_csv(ROOT / 'artifacts/mask_disagreement_subgroups.csv').round(4))
print('Post-hoc border diagnostic: remove a 16-pixel border on every side.')
display(pd.read_csv(ROOT / 'artifacts/mask_interior_summary.csv').round(5))
display(pd.read_csv(ROOT / 'artifacts/mask_interior_ci.csv').round(5))
"""),
md("""## Qualitative results
The figure explicitly labels an improvement case, a typical case and a failure case. The improvement example is selected at the 95th percentile of interior PSNR gain (excluding the 16-pixel border); the other examples use median and 5th-percentile full-frame gain. Labels report full-frame PSNR changes, including on crop figures. This post-hoc illustrative selection does not affect any aggregate metric or model selection.
"""),
code("display(Image(filename=str(ROOT / 'artifacts/mask_examples.png')))"),
md("""## Two-frame inference demo
The predictor below receives only the two endpoint images. Ground truth is loaded separately for visualization and metrics. The demo bundle contains a few held-out triplets; it is not training data.
"""),
code("""demo = np.load(ROOT / 'artifacts/demo_triplets.npz')
strength = json.loads((ROOT / 'artifacts/heuristic_selection.json').read_text())['strength']
interpolator = Interpolator(ROOT / 'runs/mask_s42/best.pt', DEVICE, strength)
frames = torch.from_numpy(demo['frames'][0]).float() / 255
result = interpolator(frames[0], frames[2])
fig, axes = plt.subplots(1, 5, figsize=(16, 3))
panels = [('Earlier', frames[0]), ('Ground truth', frames[1]), ('Later', frames[2]),
          ('Interpolated', result['learned'][0].cpu()), ('Learned mask', result['learned_mask'][0, 0].cpu())]
for ax, (title, image) in zip(axes, panels):
    ax.imshow(image.permute(1, 2, 0).numpy() if image.ndim == 3 else image.numpy(), cmap='viridis', vmin=0, vmax=1)
    ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show()
pred = result['learned'][0].cpu()
mse = (pred-frames[1]).square().mean().item()
print(f'Demo PSNR: {-10*math.log10(max(mse, 1e-12)):.3f} dB')
saved = pd.read_csv(ROOT / 'artifacts/mask_metrics_per_triplet.csv')
expected = saved[(saved.key == str(demo['keys'][0])) & (saved.method == 'mask_s42')].iloc[0].psnr
print(f'Cached evaluation PSNR: {expected:.3f} dB')
assert abs(-10*math.log10(max(mse, 1e-12))-expected) < .05, 'Investigate inference/cache mismatch'
"""),
code("display(Video(str(ROOT / 'artifacts/interpolation_demo.mp4'), embed=True))"),
md(r"""## Generative extension: residual L1 regression vs flow matching
Both refiners use the same 3.31M-parameter time-conditioned U-Net and the same frozen seed-42 mask. Their target is $(I_t-\hat I_t)/0.25$. L1 predicts the residual directly. FM learns velocity along a Gaussian-noise-to-residual path; Euler/Heun integrate it at inference.

References for this extension: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598), and [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206). We test these training/sampling ideas in a much smaller conditional residual model; their success elsewhere does not guarantee better reconstruction here.

We compare 12k-step L1/FM, EMA, logit-normal time sampling, condition dropout for CFG, and longer 30k training. The 30k runs start from scratch with a 30k learning-rate schedule. All refiners share cached full-frame conditioning and matching crop sampling. Temporal swapping is omitted in this extension because its cached mask conditioning is directional. Checkpoint monitoring and sampler diagnostics use two separate fixed noise streams; every sampler variant shares the same per-triplet diagnostic noise. This is an adaptation of the colleagues' code, not an exact rerun of its original training stream.
"""),
code("""refiners = pd.read_csv(ROOT / 'artifacts/refiner_test_summary.csv')
display(refiners[['method', 'n_triplets', 'psnr', 'ssim', 'mae', 'lpips', 'pooled_psnr_dis']].round(5))
diagnostics = pd.read_csv(ROOT / 'artifacts/refiner_validation_summary.csv')
display(diagnostics[['experiment', 'checkpoint_step', 'psnr', 'ssim', 'lpips',
                     'pooled_psnr_dis', 'refiner_ms_per_triplet']].round(5))
display(pd.read_csv(ROOT / 'artifacts/refiner_paired_ci.csv').round(5))
"""),
code("""display(Image(filename=str(ROOT / 'artifacts/learning_curves.png')))
display(Image(filename=str(ROOT / 'artifacts/refiner_examples.png')))"""),
md("""## Runtime and limitations
The small mask adds computation on top of two RAFT passes. FM additionally requires many refiner evaluations; a better ODE solver is not free. Runtime results include the hardware, resolution, batch size and measured scope. The 1.2k-triplet training subset and single refiner seed limit conclusions about scaling and stability. A mask cannot create colors/details outside the convex blend of its two warped inputs.
"""),
code("""runtime = json.loads((ROOT / 'artifacts/runtime.json').read_text())
display(pd.Series(runtime).to_frame('value'))
print((ROOT / 'artifacts/conclusions.txt').read_text())
"""),
md("""## Reproducibility and provenance
`src/` contains shared model/inference code; `scripts/` contains data preparation, training, evaluation and report generation. `runs/` preserves model configurations, histories and best checkpoints; `artifacts/` preserves per-triplet metrics and selection decisions. The two original notebooks are archived unchanged with SHA-256 checksums. See `README.md` for full commands and the distinction between original exploratory results and the corrected rerun.
"""),
]
n.metadata = dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                  language_info=dict(name="python", version="3.12"))
nbf.write(n, ROOT / "occlusion_aware_frame_interpolation_final.ipynb")
print("Notebook written")
