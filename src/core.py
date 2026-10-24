# Adapted from the supplied fps_generator_kaggle.ipynb, core cell.
"""Core of the evening experiment: forward splat to the midpoint grid, occlusion maps,
heuristic mask with the correct sign, features on the t-grid, a small mask U-Net,
region metrics with a triplet bootstrap. Pure torch, CPU/CUDA; no gradients through warps.
Coordinates: flow in pixels, (x, y) = (channel 0, channel 1); grid 0 / grid 1 / grid t.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------------- warps

def backward_warp(img, flow):
    """Sample img at p + flow(p). img [B,C,H,W], flow [B,2,H,W] px. Border padding."""
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=img.dtype),
        torch.arange(W, device=img.device, dtype=img.dtype), indexing="ij")
    gx = 2.0 * (xx[None] + flow[:, 0]) / max(W - 1, 1) - 1.0
    gy = 2.0 * (yy[None] + flow[:, 1]) / max(H - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def fb_occlusion(f_fwd, f_bwd, a1=0.01, a2=0.5):
    """UnFlow forward-backward check on the grid of f_fwd. 1 = no correspondence in the other frame."""
    fb = backward_warp(f_bwd, f_fwd)
    diff = (f_fwd + fb).square().sum(1, keepdim=True)
    mag = f_fwd.square().sum(1, keepdim=True) + fb.square().sum(1, keepdim=True)
    H, W = f_fwd.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(H, device=f_fwd.device),
                            torch.arange(W, device=f_fwd.device), indexing="ij")
    x, y = xx + f_fwd[:, 0], yy + f_fwd[:, 1]
    outside = ((x < 0) | (x > W - 1) | (y < 0) | (y > H - 1))[:, None]
    return ((diff > a1 * mag + a2) | outside).to(f_fwd.dtype)


def splat(values, flow, importance=None):
    """Forward bilinear splat of values from the source grid to p + flow.
    values [B,C,H,W], flow [B,2,H,W] px, importance [B,1,H,W] > 0 (collision weights).
    Returns num [B,C,H,W], den [B,1,H,W] (sum importance*bilinear), cnt [B,1,H,W] (sum bilinear)."""
    B, C, H, W = values.shape
    dev, dt = values.device, values.dtype
    if importance is None:
        importance = torch.ones(B, 1, H, W, device=dev, dtype=dt)
    yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=dt),
                            torch.arange(W, device=dev, dtype=dt), indexing="ij")
    x = xx[None] + flow[:, 0]
    y = yy[None] + flow[:, 1]
    x0, y0 = x.floor(), y.floor()
    pack = torch.cat([values * importance, importance, torch.ones_like(importance)], 1)  # [B,C+2,H,W]
    K = C + 2
    out = torch.zeros(K, B * H * W, device=dev, dtype=dt)
    boff = (torch.arange(B, device=dev) * H * W)[:, None, None]
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (1 - (x - xi).abs()).clamp(min=0) * (1 - (y - yi).abs()).clamp(min=0)
            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            w = w * valid
            idx = (yi.clamp(0, H - 1) * W + xi.clamp(0, W - 1)).long() + boff  # [B,H,W]
            out.index_add_(1, idx.reshape(-1), (pack * w[:, None]).permute(1, 0, 2, 3).reshape(K, -1))
    out = out.view(K, B, H, W).permute(1, 0, 2, 3)
    return out[:, :C], out[:, C:C + 1], out[:, C + 1:C + 2]


@torch.no_grad()
def make_warps(a, b, f01, f10, alpha=10.0, hole_thr=0.3):
    """Both frames splatted to the midpoint grid. Collisions weighted by brightness-constancy
    importance exp(-alpha*err). Holes (coverage < hole_thr) filled with the old backward half-flow warp.
    Returns dict on grid t: w0 w1 f01_t f10_t o0_t o1_t hole0 hole1 cnt0 cnt1."""
    f01, f10 = f01.float(), f10.float()
    err0 = (a - backward_warp(b, f01)).abs().mean(1, keepdim=True)
    err1 = (b - backward_warp(a, f10)).abs().mean(1, keepdim=True)
    o0 = fb_occlusion(f01, f10)
    o1 = fb_occlusion(f10, f01)
    num0, den0, cnt0 = splat(torch.cat([a, f01, o0], 1), 0.5 * f01, torch.exp(-alpha * err0))
    num1, den1, cnt1 = splat(torch.cat([b, f10, o1], 1), 0.5 * f10, torch.exp(-alpha * err1))
    s0 = num0 / den0.clamp_min(1e-8)
    s1 = num1 / den1.clamp_min(1e-8)
    hole0 = (cnt0 < hole_thr).float()
    hole1 = (cnt1 < hole_thr).float()
    h0, h1 = hole0.bool(), hole1.bool()
    w0 = torch.where(h0, backward_warp(a, -0.5 * f01), s0[:, :3])
    w1 = torch.where(h1, backward_warp(b, -0.5 * f10), s1[:, :3])
    return dict(
        w0=w0, w1=w1,
        f01_t=torch.where(h0, f01, s0[:, 3:5]), f10_t=torch.where(h1, f10, s1[:, 3:5]),
        o0_t=torch.where(h0, torch.zeros_like(hole0), s0[:, 5:6]),
        o1_t=torch.where(h1, torch.zeros_like(hole1), s1[:, 5:6]),
        hole0=hole0, hole1=hole1, cnt0=cnt0, cnt1=cnt1)


# ----------------------------------------------------------------------------- masks

def heuristic_mask_raw(Wd, s=1.0):
    """Before hole override. o0_t = pixel came from frame 0 and has no match in frame 1 -> trust w0 -> M up."""
    return (0.5 + 0.5 * s * (Wd["o0_t"] - Wd["o1_t"])).clamp(0, 1)


def heuristic_mask(Wd, s=1.0):
    m = heuristic_mask_raw(Wd, s)
    m = torch.where(Wd["hole0"].bool(), torch.zeros_like(m), m)   # nothing from frame 0 here -> w1
    m = torch.where(Wd["hole1"].bool(), torch.ones_like(m), m)    # nothing from frame 1 here -> w0
    m = torch.where(Wd["hole0"].bool() & Wd["hole1"].bool(), torch.full_like(m, 0.5), m)
    return m


def blend(Wd, m):
    return m * Wd["w0"] + (1 - m) * Wd["w1"]


def oracle_alpha(Wd, target):
    """Per-pixel least-squares alpha in [0,1] using the target. Upper bound, not a method."""
    d = Wd["w0"] - Wd["w1"]
    r = target - Wd["w1"]
    return ((d * r).sum(1, keepdim=True) / d.square().sum(1, keepdim=True).clamp_min(1e-8)).clamp(0, 1)


def shuffled_alpha(alpha):
    """Null: oracle alpha of another triplet (roll along batch; mirror when batch is 1)."""
    return alpha.roll(1, dims=0) if alpha.shape[0] > 1 else alpha.flip(-1)


FLOW_SCALE = 20.0
FEAT_X_CHANNELS = (9, 11)   # x-components of f01_t, f10_t: negate on horizontal flip


def make_features(Wd, heur_s=1.0):
    """18 channels on grid t."""
    return torch.cat([
        Wd["w0"], Wd["w1"], (Wd["w0"] - Wd["w1"]).abs(),
        Wd["f01_t"] / FLOW_SCALE, Wd["f10_t"] / FLOW_SCALE,
        Wd["o0_t"], Wd["o1_t"], Wd["hole0"], Wd["hole1"],
        heuristic_mask(Wd, heur_s)], 1)


def dis_region(Wd, tau=0.04):
    """Where the two warps disagree: the only place a blending mask can act. Method-independent."""
    return ((Wd["w0"] - Wd["w1"]).abs().mean(1, keepdim=True) > tau).float()


def occ_region(Wd):
    return (((Wd["o0_t"] + Wd["o1_t"]) > 0.5) | Wd["hole0"].bool() | Wd["hole1"].bool()).float()


# ----------------------------------------------------------------------------- net

def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class MaskUNet(nn.Module):
    """3-level U-Net, ~95k params. Last conv zero-init -> starts at exactly 0.5 (= fixed blend)."""
    def __init__(self, in_ch=18, w=(16, 32, 48)):
        super().__init__()
        self.e1 = _block(in_ch, w[0])
        self.e2 = _block(w[0], w[1])
        self.e3 = _block(w[1], w[2])
        self.d2 = _block(w[2] + w[1], w[1])
        self.d1 = _block(w[1] + w[0], w[0])
        self.out = nn.Conv2d(w[0], 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        e1 = self.e1(x)
        e2 = self.e2(F.avg_pool2d(e1, 2))
        e3 = self.e3(F.avg_pool2d(e2, 2))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False), e1], 1))
        return torch.sigmoid(self.out(d1))[..., :H, :W]


def hflip_features(feat):
    feat = feat.flip(-1)
    for c in FEAT_X_CHANNELS:
        feat[:, c] = -feat[:, c]
    return feat


# ----------------------------------------------------------------------------- metrics

REGIONS = ("full", "dis", "occ")


@torch.no_grad()
def region_sse(preds, target, regions):
    """preds: dict name -> [B,3,H,W]; regions: dict name -> [B,1,H,W] in {0,1}.
    Returns rows: (b, method, region, sse, n) with n = number of RGB values in the region."""
    rows = []
    B = target.shape[0]
    for rname, rm in regions.items():
        n = rm.sum(dim=(1, 2, 3)) * 3
        for mname, p in preds.items():
            se = ((p.clamp(0, 1) - target).square() * rm).sum(dim=(1, 2, 3))
            for i in range(B):
                rows.append((i, mname, rname, float(se[i]), float(n[i])))
    return rows


def pooled_mse(df, method, region):
    d = df[(df.method == method) & (df.region == region)]
    return d.sse.sum() / max(d.n.sum(), 1.0)


def bootstrap_region(df, region, methods, ref="fixed", oracle="oracle", n_boot=2000, seed=0):
    """Triplet bootstrap of delta dB vs ref and gap-closed fraction G vs oracle.
    df columns: key, method, region, sse, n. Returns dict method -> (point, lo, hi) for 'db' and 'G'."""
    sub = df[df.region == region]
    keys = sorted(sub.key.unique())
    piv_sse = sub.pivot_table(index="key", columns="method", values="sse", aggfunc="sum").loc[keys]
    piv_n = sub.pivot_table(index="key", columns="method", values="n", aggfunc="sum").loc[keys]
    S = piv_sse.to_numpy()
    N = piv_n[ref].to_numpy()
    cols = list(piv_sse.columns)
    rng = np.random.default_rng(seed)
    T = len(keys)
    idx = np.vstack([np.arange(T)[None], rng.integers(0, T, size=(n_boot, T))])  # row 0 = point estimate

    def stats(ix):
        s = S[ix].sum(0)
        n = N[ix].sum()
        mse = {c: s[j] / max(n, 1.0) for j, c in enumerate(cols)}
        out = {}
        for m in methods:
            db = 10 * math.log10(max(mse[ref], 1e-12) / max(mse[m], 1e-12))
            gap = mse[ref] - mse.get(oracle, 0.0)
            G = (mse[ref] - mse[m]) / gap if gap > 1e-12 else float("nan")
            out[m] = (db, G)
        return out

    all_stats = [stats(ix) for ix in idx]
    res = {}
    for m in methods:
        dbs = np.array([s[m][0] for s in all_stats])
        Gs = np.array([s[m][1] for s in all_stats])
        res[m] = {
            "db": (dbs[0], np.percentile(dbs[1:], 2.5), np.percentile(dbs[1:], 97.5)),
            "G": (Gs[0], np.nanpercentile(Gs[1:], 2.5), np.nanpercentile(Gs[1:], 97.5)),
        }
    return res


def psnr_from_mse(mse):
    return -10 * math.log10(max(mse, 1e-12))