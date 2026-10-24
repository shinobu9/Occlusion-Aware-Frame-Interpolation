"""Use the same per-key noise as evaluation for visible refinement comparisons."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.refiner import RefinerUNet, RES_SCALE, l1_predict, best_dis_window
from src.runtime import setup
from scripts.train_refiner import sample
from scripts.evaluate_refiner import noise_for_keys


def main():
    setup(42)
    root = Path(__file__).resolve().parents[1]
    art = root / "artifacts"
    curves, axes_curve = plt.subplots(1, 2, figsize=(13, 4))
    for ax, names, title in [
        (axes_curve[0], ["mask_s42", "mask_s43", "mask_s44"], "Mask validation (150 triplets)"),
        (axes_curve[1], ["l1", "fm", "fm_30k", "fm_ema", "fm_ema_lognormal_30k"], "Refiner validation (64 triplets)")]:
        for name in names:
            records = [json.loads(line) for line in (root / "runs" / name / "history.jsonl").read_text().splitlines()]
            records = [r for r in records if "val" in r]
            ax.plot([r["step"] for r in records], [r["val"]["dis"] for r in records], label=name)
        ax.set(title=title, xlabel="Optimizer step", ylabel="Pooled disagreement PSNR (dB)")
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    curves.tight_layout()
    curves.savefig(art / "learning_curves.png", dpi=150)
    plt.close(curves)
    cases = json.loads((art / "example_selection.json").read_text())
    manifest = json.loads((root / "data/test_manifest.json").read_text())
    indices = {r["key"]: i for i, r in enumerate(manifest)}
    conditions = np.load(root / "data/test_condition.npy", mmap_mode="r")
    frames = np.load(root / "data/test_frames.npy", mmap_mode="r")
    selection = json.loads((art / "final_selection.json").read_text())["config"]
    def load(path):
        model = RefinerUNet().cuda().eval()
        model.load_state_dict(torch.load(root / path / "best.pt", map_location="cuda", weights_only=False)["model"])
        return model
    l1, fm, selected = load("runs/l1"), load("runs/fm"), load(selection["run"])
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    with torch.inference_mode():
        for row, case in enumerate(cases):
            idx, key = indices[case["key"]], case["key"]
            c = torch.from_numpy(np.array(conditions[idx:idx+1])).cuda().float()
            gt = torch.from_numpy(np.array(frames[idx:idx+1, 1])).cuda().float()/255
            bl = c[:, :3]
            pl1 = bl+RES_SCALE*l1_predict(l1, c)
            pfm = bl+RES_SCALE*sample(fm, c, noise=noise_for_keys([key]))
            psel = bl+RES_SCALE*sample(selected, c, steps=selection["steps"], solver=selection["solver"],
                                      guidance=selection["guidance"], noise=noise_for_keys([key]))
            y, x = best_dis_window(c[:, -1:])[0]
            for col, (title, image) in enumerate(zip(["Ground truth", "Mask only", "+ L1", "+ FM", "+ selected FM", "Warp disagreement"], [gt, bl, pl1, pfm, psel, c[:, -1:]])):
                im = image[0, :, y:y+128, x:x+128].clamp(0, 1).cpu()
                axes[row, col].imshow(im.permute(1, 2, 0).numpy() if im.shape[0] == 3 else im[0].numpy(), cmap="gray", vmin=0, vmax=1)
                axes[row, col].set_title(title, fontsize=11)
                axes[row, col].axis("off")
            axes[row, 0].text(0, -.08, key, transform=axes[row, 0].transAxes, fontsize=10)
    fig.tight_layout()
    fig.savefig(art / "refiner_examples.png", dpi=150, bbox_inches="tight")
    print("Saved refiner examples")


if __name__ == "__main__":
    main()
