"""Meaningful geometry and metric checks before spending GPU time."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.core import splat, make_warps, blend, heuristic_mask, oracle_alpha, MaskUNet
from src.data import swap_features
from src.metrics import ssim
from skimage.metrics import structural_similarity
import numpy as np

torch.manual_seed(7)
a = torch.rand(2, 3, 16, 24)
f = torch.zeros(2, 2, 16, 24)
num, den, cnt = splat(a, f)
assert torch.allclose(num/den, a)
f[:, 0] = 2
num, den, cnt = splat(a, f)
assert torch.allclose((num/den.clamp_min(1e-8))[..., 2:], a[..., :-2])
f.zero_()
wd = make_warps(a, a, f, f)
assert torch.allclose(blend(wd, heuristic_mask(wd)), a)
wd["hole0"].fill_(1)
wd["hole1"].fill_(1)
assert (heuristic_mask(wd) == .5).all()
wd["w1"] = torch.rand_like(a)
t = torch.rand_like(a)
best = blend(wd, oracle_alpha(wd, t))
for alpha in (0., .25, .5, .75, 1.):
    assert ((best-t).square().sum(1) <= (blend(wd, alpha)-t).square().sum(1)+1e-6).all()
feat = torch.rand(2, 18, 16, 24)
assert torch.allclose(swap_features(swap_features(feat)), feat)
model = MaskUNet()
mask = model(feat)
assert (mask == .5).all()
loss = (mask*feat[:, :3]+(1-mask)*feat[:, 3:6]-t).square().mean()
loss.backward()
assert model.out.weight.grad.abs().sum() > 0
x, y = torch.rand(3, 3, 32, 40), torch.rand(3, 3, 32, 40)
x[0].fill_(.2)
y[0].fill_(.2)
reference = [structural_similarity(a.permute(1, 2, 0).numpy(), b.permute(1, 2, 0).numpy(),
                                  channel_axis=2, data_range=1.) for a, b in zip(x, y)]
assert np.allclose(ssim(x, y).numpy(), reference, atol=1e-5)
print("PASS: identity, translation, double holes, oracle bound, temporal symmetry, gradient, skimage SSIM parity")
