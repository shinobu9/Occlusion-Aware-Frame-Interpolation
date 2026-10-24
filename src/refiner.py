# Adapted from the supplied fps_generator_kaggle.ipynb, refiner cell.
"""Conditional flow-matching refiner for the residual target - blend on the midpoint grid.
Same U-Net doubles as a deterministic L1 regressor (x = 0, t = 0) for the regression-vs-generative control.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

RES_SCALE = 0.25          # residual / RES_SCALE ~ unit scale inside dis
COND_CH = 15              # blend 3, w0 3, w1 3, mask 1, hole0 hole1 o0_t o1_t 4, dis 1


def refiner_cond(Wd, mask, blend, dis):
    return torch.cat([blend, Wd["w0"], Wd["w1"], mask, Wd["hole0"], Wd["hole1"], Wd["o0_t"], Wd["o1_t"], dis], 1)


class TimeEmb(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):                       # t [B] in [0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        ang = t.float()[:, None] * 1000.0 * freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], 1))


class Block(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout)
        self.norm = nn.GroupNorm(8, cout)

    def forward(self, x, te):
        h = F.silu(self.c1(x) + self.t(te)[:, :, None, None])
        return F.silu(self.norm(self.c2(h)))


class RefinerUNet(nn.Module):
    """3 levels + bottleneck, 3,306,243 params at widths (64,128,192). Pads to /8."""
    def __init__(self, cond_ch=COND_CH, w=(64, 128, 192), tdim=128):
        super().__init__()
        self.temb = TimeEmb(tdim)
        self.e1 = Block(3 + cond_ch, w[0], tdim)
        self.e2 = Block(w[0], w[1], tdim)
        self.e3 = Block(w[1], w[2], tdim)
        self.mid = Block(w[2], w[2], tdim)
        self.d3 = Block(w[2] + w[2], w[2], tdim)
        self.d2 = Block(w[2] + w[1], w[1], tdim)
        self.d1 = Block(w[1] + w[0], w[0], tdim)
        self.out = nn.Conv2d(w[0], 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 8, (-W) % 8
        h = F.pad(torch.cat([x, cond], 1), (0, pw, 0, ph), mode="replicate")
        te = self.temb(t)
        up = lambda z: F.interpolate(z, scale_factor=2, mode="bilinear", align_corners=False)
        e1 = self.e1(h, te)
        e2 = self.e2(F.avg_pool2d(e1, 2), te)
        e3 = self.e3(F.avg_pool2d(e2, 2), te)
        m = self.mid(F.avg_pool2d(e3, 2), te)
        d3 = self.d3(torch.cat([up(m), e3], 1), te)
        d2 = self.d2(torch.cat([up(d3), e2], 1), te)
        d1 = self.d1(torch.cat([up(d2), e1], 1), te)
        return self.out(d1)[..., :H, :W]


def fm_loss(net, x1, cond):
    """Rectified flow: x_t = (1-t) x0 + t x1, target velocity x1 - x0."""
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device)
    xt = (1 - t)[:, None, None, None] * x0 + t[:, None, None, None] * x1
    return F.mse_loss(net(xt, t, cond), x1 - x0)


def l1_loss(net, x1, cond):
    """Deterministic control: same net, no noise, predicts the residual directly."""
    pred = net(torch.zeros_like(x1), torch.zeros(x1.shape[0], device=x1.device), cond)
    return (pred - x1).abs().mean()


@torch.no_grad()
def fm_sample(net, cond, steps=20, generator=None):
    B, _, H, W = cond.shape
    x = torch.randn(B, 3, H, W, device=cond.device, generator=generator)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=cond.device)
        x = x + net(x, t, cond) / steps
    return x


@torch.no_grad()
def l1_predict(net, cond):
    B = cond.shape[0]
    return net(torch.zeros(B, 3, *cond.shape[-2:], device=cond.device), torch.zeros(B, device=cond.device), cond)


def dis_centered_crops(tensors, dis, crop, rng):
    """Crop each sample around a random pixel of its dis map (random anywhere if dis is empty).
    tensors: list of [B,C,H,W]; returns list of [B,C,crop,crop]."""
    B, _, H, W = dis.shape
    outs = [[] for _ in tensors]
    for b in range(B):
        ys, xs = torch.nonzero(dis[b, 0], as_tuple=True)
        if len(ys) > 0:
            k = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[k]), int(xs[k])
        else:
            cy, cx = int(rng.integers(0, H)), int(rng.integers(0, W))
        y0 = min(max(cy - crop // 2, 0), H - crop)
        x0 = min(max(cx - crop // 2, 0), W - crop)
        for o, tsr in zip(outs, tensors):
            o.append(tsr[b:b + 1, :, y0:y0 + crop, x0:x0 + crop])
    return [torch.cat(o, 0) for o in outs]


@torch.no_grad()
def best_dis_window(dis, size=128, stride=16):
    """Per sample: top-left (y0, x0) of the size x size window with the most dis pixels."""
    dens = F.avg_pool2d(dis, size, stride=stride)          # [B,1,h,w]
    B, _, h, w = dens.shape
    flat = dens.view(B, -1).argmax(1)
    return [(int(i // w) * stride, int(i % w) * stride) for i in flat.tolist()]
