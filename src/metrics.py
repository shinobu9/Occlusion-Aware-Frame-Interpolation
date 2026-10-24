"""Per-image RGB metrics on [0,1]. SSIM matches skimage's default 7x7 window."""
import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def ssim(x, y):
    ux, uy = F.avg_pool2d(x, 7, 1), F.avg_pool2d(y, 7, 1)
    vx = (F.avg_pool2d(x*x, 7, 1)-ux*ux)*(49/48)
    vy = (F.avg_pool2d(y*y, 7, 1)-uy*uy)*(49/48)
    cov = (F.avg_pool2d(x*y, 7, 1)-ux*uy)*(49/48)
    score = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return score.mean((1, 2, 3))


def boundary_region(features):
    flow = features[:, 9:13]*20
    dx = F.pad((flow[..., 1:]-flow[..., :-1]).square().sum(1, keepdim=True).sqrt(), (0, 1, 0, 0))
    dy = F.pad((flow[..., 1:, :]-flow[..., :-1, :]).square().sum(1, keepdim=True).sqrt(), (0, 0, 0, 1))
    return ((dx+dy) > 1.).float()


@torch.inference_mode()
def measure(pred, target, regions):
    pred = pred.clamp(0, 1)
    err = pred-target
    mse = err.square().mean((1, 2, 3))
    values = dict(psnr=-10*mse.clamp_min(1e-12).log10(), ssim=ssim(pred, target),
                  mae=err.abs().mean((1, 2, 3)), mse=mse)
    for name, mask in regions.items():
        values[f"sse_{name}"] = (err.square()*mask).sum((1, 2, 3))
        values[f"n_{name}"] = mask.sum((1, 2, 3))*3
    return {k: v.cpu().numpy() for k, v in values.items()}


def summarize(df):
    out = []
    for method, d in df.groupby("method", sort=False):
        row = dict(method=method, n_triplets=len(d))
        for k in ("psnr", "ssim", "mae", "mse", "lpips", "lpips_window"):
            if k in d:
                row[k] = float(d[k].mean())
        for k in d.columns:
            if k.startswith("sse_"):
                region = k[4:]
                denom = d[f"n_{region}"].sum()
                row[f"pooled_psnr_{region}"] = float(-10*np.log10(max(d[k].sum()/denom, 1e-12))) if denom else None
        out.append(row)
    return out
