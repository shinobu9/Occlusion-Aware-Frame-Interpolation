"""Matched L1/FM experiments plus separately identified EMA, time-density and CFG ablations."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from src.refiner import RefinerUNet, RES_SCALE, dis_centered_crops, l1_predict
from src.runtime import setup, atomic_save, save_config, rng_state, restore_rng


@torch.inference_mode()
def sample(model, cond, steps=20, solver="euler", generator=None, guidance=1., noise=None):
    x = torch.randn(len(cond), 3, *cond.shape[-2:], device=cond.device, generator=generator) if noise is None else noise.clone()
    def velocity(x, t):
        v = model(x, t, cond)
        if guidance != 1.:
            u = model(x, t, torch.zeros_like(cond))
            v = u + guidance*(v-u)
        return v
    for i in range(steps):
        t = torch.full((len(cond),), i/steps, device=cond.device)
        v = velocity(x, t)
        y = x+v/steps
        if solver == "heun":
            v1 = velocity(y, torch.full_like(t, (i+1)/steps))
            y = x+(v+v1)/(2*steps)
        x = y
    return x


@torch.inference_mode()
def validate(model, mode, cond, target):
    model.eval()
    sse = np.zeros(2)
    count = np.zeros(2)
    for i in range(0, len(cond), 4):
        c, t = cond[i:i+4].float(), target[i:i+4].float()/255
        gen = torch.Generator(device="cuda").manual_seed(19000+i)
        r = l1_predict(model, c) if mode == "l1" else sample(model, c, generator=gen)
        pred = (c[:, :3]+RES_SCALE*r).clamp(0, 1)
        se = (pred-t).square().sum(1, keepdim=True)
        masks = torch.cat((torch.ones_like(c[:, -1:]), c[:, -1:]), 1)
        sse += (se*masks).sum((0, 2, 3)).cpu().numpy()
        count += (3*masks.sum((0, 2, 3))).cpu().numpy()
    return dict(zip(("full", "dis"), (-10*np.log10(sse/count)).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--mode", choices=["fm", "l1"], default="fm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--val-count", type=int, default=64)
    ap.add_argument("--ema", type=float, default=0.)
    ap.add_argument("--time-density", choices=["uniform", "logit_normal"], default="uniform")
    ap.add_argument("--cfg-dropout", type=float, default=0.)
    args = ap.parse_args()
    if (args.run / "DONE.json").exists():
        print("already complete", args.run, flush=True)
        return
    setup(args.seed)
    save_config(args, args.run)
    # Static conditioning fits comfortably on one H100; avoid recopying full frames every step.
    train_c = torch.from_numpy(np.load(args.data / "train_condition.npy")).cuda()
    train_t = torch.from_numpy(np.array(np.load(args.data / "train_frames.npy", mmap_mode="r")[:, 1])).cuda()
    val_c = torch.from_numpy(np.load(args.data / "val_condition.npy")[:args.val_count]).cuda()
    val_t = torch.from_numpy(np.array(np.load(args.data / "val_frames.npy", mmap_mode="r")[:args.val_count, 1])).cuda()
    model = RefinerUNet().cuda()
    ema = copy.deepcopy(model).eval() if args.ema else None
    if ema is not None:
        ema.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    rng = np.random.default_rng(args.seed+1)
    start_step, best = 0, -float("inf")
    if (args.run / "latest.pt").exists():
        ck = torch.load(args.run / "latest.pt", map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        start_step, best = ck["step"], ck["best"]
        restore_rng(ck["rng"], rng)
    start = time.monotonic()
    print("training", vars(args), "GPU allocated GB", torch.cuda.memory_allocated()/1e9, flush=True)
    with (args.run / "history.jsonl").open("a") as log:
        for step in range(start_step+1, args.steps+1):
            ids = torch.as_tensor(rng.choice(len(train_c), args.batch, replace=False), device="cuda")
            c = train_c[ids].float()
            target = train_t[ids].float()/255
            x1 = (target-c[:, :3])/RES_SCALE
            c, x1 = dis_centered_crops([c, x1], c[:, -1:], args.crop, rng)
            if rng.random() < .5:
                c, x1 = c.flip(-1), x1.flip(-1)
            # Temporal swapping is omitted: the frozen mask condition is directional.
            model.train()
            if args.mode == "l1":
                loss = (l1_predict_train(model, c)-x1).abs().mean()
            else:
                noise = torch.randn_like(x1)
                t = torch.rand(len(c), device="cuda") if args.time_density == "uniform" else torch.randn(len(c), device="cuda").sigmoid()
                xt = (1-t[:, None, None, None])*noise+t[:, None, None, None]*x1
                if args.cfg_dropout:
                    keep = (torch.rand(len(c), 1, 1, 1, device="cuda") >= args.cfg_dropout)
                    c = c*keep
                loss = F.mse_loss(model(xt, t, c), x1-noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for ep, p in zip(ema.parameters(), model.parameters()):
                        ep.lerp_(p, 1-args.ema)
            if step % 100 == 0 or step == args.steps:
                elapsed = time.monotonic()-start
                rec = dict(step=step, loss=loss.item(), seconds=elapsed,
                           eta_seconds=elapsed/(step-start_step)*(args.steps-step))
                if step % args.eval_every == 0 or step == args.steps:
                    eval_model = ema if ema is not None else model
                    scores = validate(eval_model, args.mode, val_c, val_t)
                    rec["val"] = scores
                    if scores["dis"] > best:
                        best = scores["dis"]
                        atomic_save(dict(model=eval_model.state_dict(), step=step, val=scores,
                                         ema=bool(ema is not None)), args.run / "best.pt")
                log.write(json.dumps(rec)+"\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step % 500 == 0 or step == args.steps:
                atomic_save(dict(model=model.state_dict(), ema=ema.state_dict() if ema is not None else None,
                                 optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                                 step=step, best=best, rng=rng_state(rng)), args.run / "latest.pt")
    (args.run / "DONE.json").write_text(json.dumps(dict(best_val_psnr_dis=best, steps=args.steps, seconds=time.monotonic()-start)))


def l1_predict_train(model, cond):
    return model(torch.zeros(len(cond), 3, *cond.shape[-2:], device=cond.device),
                 torch.zeros(len(cond), device=cond.device), cond)


if __name__ == "__main__":
    main()
