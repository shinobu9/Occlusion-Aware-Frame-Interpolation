"""Publish three explained tables from evaluation outputs (artifacts/ is untracked)."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / 'artifacts', ROOT / 'results'
OUT.mkdir(exist_ok=True)
mask = pd.read_csv(ART / 'mask_summary.csv').assign(family='mask')
ref = pd.read_csv(ART / 'refiner_test_summary.csv')
ref = ref[ref.method.isin(['learned_cached_reference', 'l1', 'fm', 'fm_mean4'])].assign(family='refiner')
labels = {
    'fixed': 'Baseline 1: fixed 50/50', 'heuristic_0.5': 'Baseline 2: flow-consistency heuristic',
    'mask_s42': 'Our mask: primary seed 42', 'mask_s43': 'Our mask: robustness seed 43',
    'mask_s44': 'Our mask: robustness seed 44', 'oracle': 'GT-informed convex-blending ceiling',
    'old_half_flow': 'Geometry control: backward half-flow', 'unwarped': 'Control: no motion compensation',
    'learned_cached_reference': 'Frozen mask reference for refiners (fp16 cache)',
    'l1': 'Mask + L1 residual refiner', 'fm': 'Mask + FM, Euler 20', 'fm_mean4': 'Mask + FM, mean of four samples',
}
test = pd.concat([mask, ref], ignore_index=True)
test.insert(2, 'description', test.method.map(labels))
columns = ['family', 'method', 'description', 'n_triplets', 'psnr', 'ssim', 'mae', 'mse', 'lpips', 'lpips_window',
           'pooled_psnr_dis', 'pooled_psnr_occ', 'pooled_psnr_boundary']
test[columns].to_csv(OUT / 'test_metrics.csv', index=False)
val = pd.read_csv(ART / 'refiner_validation_summary.csv')
descriptions = {
    'val_l1': 'L1 control; 12k training steps',
    'val_fm_ema': 'FM 12k + EMA 0.999',
    'val_fm_ema_lognormal': 'FM 12k + EMA + logit-normal time',
    'val_fm_ema_lognormal_cfg': 'FM 12k + EMA + logit-normal + condition dropout 0.1; guidance 1',
    'val_cfg_w2': 'Same condition-dropout checkpoint; guidance 2',
    'val_cfg_w3': 'Same condition-dropout checkpoint; guidance 3',
    'val_fm_30k': 'Baseline FM trained from scratch for 30k steps',
    'val_fm_ema_lognormal_30k': 'FM + EMA + logit-normal trained from scratch for 30k steps',
}
for solver in ('euler', 'heun'):
    for steps in (20, 50, 100):
        descriptions[f'val_fm_{solver}{steps}'] = f'Same baseline FM checkpoint; {solver} {steps} sampling steps'
val.insert(1, 'description', val.experiment.map(descriptions))
assert val.description.notna().all()
val.to_csv(OUT / 'validation_ablations.csv', index=False)
parts = [pd.read_csv(ART / 'mask_paired_ci.csv').assign(family='mask', scope='full')]
parts.append(pd.read_csv(ART / 'mask_regional_ci.csv').rename(columns={'region':'scope'}).assign(family='mask', method='mask_s42', metric='psnr'))
parts.append(pd.read_csv(ART / 'mask_interior_ci.csv').assign(family='mask', scope='interior_16px', method='mask_s42'))
rci = pd.read_csv(ART / 'refiner_paired_ci.csv')
rci = rci[rci.experiment != 'test_selected_fm'].copy()
rci['method'] = rci.experiment.str.removeprefix('test_')
parts.append(rci.assign(family='refiner', scope='full', baseline='learned_cached_reference'))
comparisons = pd.concat(parts, ignore_index=True)
comparisons['ci_level'] = .95
comparisons[['family','scope','method','baseline','metric','delta','ci_lo','ci_hi','ci_level']].to_csv(OUT / 'paired_comparisons.csv', index=False)
read_json = lambda name: json.loads((ART / name).read_text())
per = pd.read_csv(ART / 'mask_metrics_per_triplet.csv')
keys = np.load(ROOT / 'demo/demo_triplets.npz')['keys'].tolist()
source_report = json.loads((ROOT / 'data/data_report.json').read_text())
data_report = dict(train_source='https://huggingface.co/datasets/bijinc/vimeo-90k-medium',
                   test_source='https://huggingface.co/datasets/paschalc24/vimeo-90k-test-full',
                   counts=source_report['counts'],
                   split_rule='Fixed sequence IDs from the published train/val/test lists',
                   limitation=source_report['limitation'], frame_order=source_report['frame_order'])
primary = mask.set_index('method')
refiners = ref.set_index('method')
gain = primary.loc['mask_s42','psnr']-primary.loc['fixed','psnr']
interpretation = (f"The primary learned mask improves mean test PSNR over fixed blending by {gain:+.3f} dB. "
                  f"L1 / FM / FM mean-of-four reach {refiners.loc['l1','psnr']:.3f} / "
                  f"{refiners.loc['fm','psnr']:.3f} / {refiners.loc['fm_mean4','psnr']:.3f} dB. "
                  "See results/README.md for experiment definitions, metric conventions and limitations.")
study = dict(data=data_report, runtime=read_json('runtime.json'),
             heuristic_strength=read_json('heuristic_selection.json')['strength'],
             disagreement_subgroups=pd.read_csv(ART / 'mask_disagreement_subgroups.csv').to_dict('records'),
             demo_psnr=per[(per.method=='mask_s42') & per.key.isin(keys)].set_index('key').psnr.to_dict(),
             interpretation=interpretation)
(OUT / 'study.json').write_text(json.dumps(study, indent=2))
print('Wrote test_metrics.csv, validation_ablations.csv, paired_comparisons.csv, study.json')
