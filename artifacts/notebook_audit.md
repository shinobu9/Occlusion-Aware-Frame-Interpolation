# Audit of the supplied notebooks

The original files are preserved in `originals/`. Their saved outputs are exploratory evidence, not the final results of this project. No reusable checkpoints or metric CSV files were present in the supplied directory, so all reported final experiments were trained and evaluated again.

| Finding | Consequence | Corrected study |
|---|---|---|
| The first notebook uses approximate backward half-flow warping; the second uses forward splatting. | Their numbers do not isolate the blending model. | Shared forward splats for all three primary methods; approximate half-flow retained as a separate geometry baseline. |
| Public previews from the original training source match test sequence `00001/0389` exactly. | The original data-loading protocol can leak test examples into training. | New pinned training source; ID and decoded-frame SHA-256 exclusion against the entire test set. |
| A heuristic function captures the default strength before the later tuning assignment. | Changing the global strength need not change the function's default. | Pass strength explicitly; fix the learned feature channel's strength at 1.0. |
| The final second-notebook comparison omits the heuristic baseline. | The stated three-way research question is not fully evaluated. | Report fixed, validation-tuned heuristic and learned masks on the same 3,782 triplets. |
| Out-of-image consistency checks and double-hole blending need explicit handling. | Invalid samples can appear consistent; fallback can favor one endpoint. | Flag out-of-frame flow mappings; use symmetric blending when both splats lack coverage. |
| The second notebook does not report SSIM in its final comparison. | One promised metric is missing. | Compute RGB SSIM for every method; verify against scikit-image. The first notebook already contains SSIM. |
| Some LPIPS confidence intervals cross zero but receive an improvement verdict. | The conclusion is stronger than its uncertainty supports. | Call such intervals inconclusive; preserve paired deltas and interval bounds. |
| FM sampling noise varies and the best validation checkpoint is not preserved. | Solver comparisons and final-checkpoint comparisons are confounded. | Save best validation checkpoints; use fixed evaluation noise per triplet across sampler settings. |
| Refiner parameter count in text is approximately 1.8M. | Model cost is understated. | Count actual parameters: 3,306,243. MaskUNet has 95,185 parameters. |
| Disagreement and flow-consistency regions are called occlusions. | These masks do not establish true visibility. | Label them diagnostic proxies throughout the report. |

## What the saved outputs showed

- First notebook: mean test PSNR approximately 33.063 / 32.678 / 33.071 dB for fixed / heuristic / learned; the learned mask is nearly constant. Execution order is not a clean end-to-end provenance record.
- Second notebook: pooled PSNR approximately 30.813 / 31.091 / 31.470 / 29.952 dB for fixed / learned / L1 / FM. This metric differs from mean per-image PSNR.
- These old numbers use different geometry, data subsets and evaluation conventions. They must not be placed alongside the corrected results as directly comparable measurements.

## Evidence and limits of the overlap check

`original_source_overlap_preview.json` records exact equality for training rows 1164–1166 (triplet index 388) and the three frames of test sequence `00001/0389` in the inspected public-source previews. It proves overlap in the inspected source snapshot, not the precise contents of an unavailable historical training cache.

The corrected split removes exact overlaps, but directory-prefix grouping is not proof of original-video independence. Similar scenes can remain, and triplet-bootstrap confidence intervals do not account for unknown video clusters.

## Deliberate experimental changes

This is an adapted rerun, not an exact reproduction of either notebook's original trajectory. It uses a clean training subset, shared cached warps, full-test evaluation, independent post-warp crops, three mask seeds and fixed-noise validation. Refiner conditioning is cached from the primary mask; temporal swapping is omitted because this conditioning is directional. Longer FM runs start from scratch with a 30k learning-rate schedule. These choices and limitations are preserved in the notebook and README.
