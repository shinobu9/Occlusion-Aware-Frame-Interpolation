"""Train the colleague's 95k mask U-Net on the corrected shared data protocol."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.core import MaskUNet, hflip_features
from src.data import CachedData, swap_features
from src.runtime import setup, atomic_save, save_config, rng_state, restore_rng


@torch.inference_mode()
def validate(model, data):
    model.eval()
    sse, count = np.zeros(3), np.zeros(3)
    mask_std = []
    for i in range(0, len(data), 8):
        f, t, reg = data.batch(np.arange(i, min(i+8, len(data))))
        m = model(f)
        p = m*f[:, :3] + (1-m)*f[:, 3:6]
        se = (p-t).square()
        masks = torch.cat((torch.ones_like(reg[:, :1]), reg), 1)
        sse += (se.sum(1, keepdim=True)*masks).sum((0, 2, 3)).cpu().numpy()
        count += (masks.sum((0, 2, 3))*3).cpu().numpy()
        mask_std.extend(m.flatten(1).std(1).cpu().tolist())
    return dict(zip(("full", "dis", "occ"), (-10*np.log10(sse/count)).tolist())), float(np.mean(mask_std))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-dis", type=float, default=4.)
    args = ap.parse_args()
    if (args.run / "DONE.json").exists():
        print("already complete", args.run, flush=True)
        return
    setup(args.seed)
    save_config(args, args.run)
    train, val = CachedData(args.data, "train"), CachedData(args.data, "val")
    model = MaskUNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(args.seed)
    start_step, best = 0, -float("inf")
    if (args.run / "latest.pt").exists():
        ck = torch.load(args.run / "latest.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_step, best = ck["step"], ck["best"]
        restore_rng(ck["rng"], rng)
    else:
        scores, std = validate(model, val)
        best = scores["dis"]
        atomic_save(dict(model=model.state_dict(), step=0, val=scores), args.run / "best.pt")
        print("initial validation", scores, flush=True)
    start = time.monotonic()
    with (args.run / "history.jsonl").open("a") as log:
        for step in range(start_step+1, args.steps+1):
            model.train()
            ids = rng.choice(len(train), args.batch, replace=False)
            f, t, reg = train.batch(ids, crop=args.crop, rng=rng)
            if rng.random() < .5:
                f = swap_features(f)
            if rng.random() < .5:
                f, t, reg = hflip_features(f), t.flip(-1), reg.flip(-1)
            m = model(f)
            pred = m*f[:, :3]+(1-m)*f[:, 3:6]
            w = 1+args.lambda_dis*reg[:, :1]
            loss = (w*(pred-t).abs()).sum()/(3*w.sum())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 100 == 0 or step == args.steps:
                elapsed = time.monotonic()-start
                rec = dict(step=step, loss=loss.item(), mask_mean=m.mean().item(), mask_std=m.std().item(),
                           seconds=elapsed, eta_seconds=elapsed/(step-start_step)*(args.steps-step))
                if step % args.eval_every == 0 or step == args.steps:
                    scores, std = validate(model, val)
                    rec.update(val=scores, val_mask_std=std)
                    if scores["dis"] > best:
                        best = scores["dis"]
                        atomic_save(dict(model=model.state_dict(), step=step, val=scores), args.run / "best.pt")
                log.write(json.dumps(rec)+"\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step % 500 == 0 or step == args.steps:
                atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                                 step=step, best=best, rng=rng_state(rng)), args.run / "latest.pt")
    (args.run / "DONE.json").write_text(json.dumps(dict(best_val_psnr_dis=best, steps=args.steps, seconds=time.monotonic()-start)))


if __name__ == "__main__":
    main()
