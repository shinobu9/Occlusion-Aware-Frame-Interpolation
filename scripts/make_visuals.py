"""Render explicitly selected success/typical/failure examples and a triplet demo."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio
from src.core import MaskUNet, heuristic_mask, blend
from src.data import CachedData, wd_from_features
from src.refiner import best_dis_window

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def main():
    torch.set_num_threads(4)
    data = CachedData(ROOT / "data", "test")
    table = pd.read_csv(ART / "mask_metrics_per_triplet.csv")
    pivot = table.pivot(index="key", columns="method", values="psnr")
    gain = (pivot.mask_s42-pivot.fixed).sort_values()
    interior = pd.read_csv(ART / "mask_interior_per_triplet.csv").pivot(index="key", columns="method", values="psnr")
    interior_gain = (interior.learned-interior.fixed).sort_values()
    chosen = [("Improvement case (95th percentile interior gain)", interior_gain.index[int(.95*(len(interior_gain)-1))]),
              ("Typical case (median gain)", gain.index[len(gain)//2]),
              ("Failure case (5th percentile gain)", gain.index[int(.05*(len(gain)-1))])]
    key_index = {r["key"]: i for i, r in enumerate(data.manifest)}
    model = MaskUNet().eval()
    model.load_state_dict(torch.load(ROOT / "runs/mask_s42/best.pt", map_location="cpu", weights_only=False)["model"])
    strength = json.loads((ART / "heuristic_selection.json").read_text())["strength"]
    fig = plt.figure(figsize=(19, 7))
    demo, video = [], []
    font = ImageFont.truetype(findfont("DejaVu Sans"), 22)
    small_font = ImageFont.truetype(findfont("DejaVu Sans"), 14)
    with torch.inference_mode():
        for row, (label, key) in enumerate(chosen):
            idx = key_index[key]
            f, t, reg = data.batch([idx], device="cpu")
            frames = np.array(data.frames[idx])
            demo.append(frames)
            wd = wd_from_features(f)
            m = model(f)
            fixed, heur, learned = blend(wd, .5), blend(wd, heuristic_mask(wd, strength)), blend(wd, m)
            rgb = lambda x: x[0].permute(1, 2, 0).numpy().clip(0, 1)
            panels = [frames[0].transpose(1, 2, 0)/255, rgb(t), frames[2].transpose(1, 2, 0)/255,
                      rgb(fixed), rgb(heur), rgb(learned), m[0, 0].numpy(), (learned-t).abs().mean(1)[0].numpy()]
            titles = ["Earlier", "Ground truth", "Later", "Fixed", "Heuristic", "Learned", "Learned mask", "Learned |error|"]
            yrow = .70-row*.30
            fig.text(.015, yrow+.255, f"{label} · {key} · ΔPSNR={gain[key]:+.3f} dB", fontsize=12)
            for col, (panel, title) in enumerate(zip(panels, titles)):
                ax = fig.add_axes((.015+col*.123, yrow, .115, .23))
                ax.imshow(panel, cmap="magma" if col == 7 else "viridis", vmin=0, vmax=.12 if col == 7 else 1)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
            # Four-panel crop makes individual mask behaviour legible on a slide.
            y0, x0 = best_dis_window(reg[:, :1])[0]
            sub, axs = plt.subplots(1, 4, figsize=(12, 3.3))
            for ax, panel, title in zip(axs, [rgb(t), rgb(fixed), rgb(learned), m[0, 0].numpy()], ["Ground truth", "Fixed", "Learned", "Mask: 0=later, 1=earlier"]):
                ax.imshow(panel[y0:y0+128, x0:x0+128], cmap="viridis", vmin=0, vmax=1); ax.set_title(title); ax.axis("off")
            sub.suptitle(f"{label} · {key} · full-frame ΔPSNR {gain[key]:+.3f} dB · disagreement crop", fontsize=10)
            sub.tight_layout()
            sub.savefig(ART / f"mask_case_{row}.png", dpi=160, bbox_inches="tight")
            plt.close(sub)
            preds = [(frames[0].transpose(1, 2, 0),)*3,
                     tuple((rgb(x)*255).round().astype(np.uint8) for x in [fixed, learned, t]),
                     (frames[2].transpose(1, 2, 0),)*3]
            for phase, images in enumerate(preds):
                canvas = Image.new("RGB", (1344, 336), "white")
                draw = ImageDraw.Draw(canvas)
                for k, (name, im) in enumerate(zip(["Fixed 50/50", "Learned mask", "Ground truth"], images)):
                    canvas.paste(Image.fromarray(im), (k*448, 50))
                    draw.text((k*448+12, 10), name, fill="black", font=font)
                draw.text((12, 312), f"{key} | {'input endpoint' if phase != 1 else 'interpolated midpoint'} | three-frame loop, slowed for inspection", fill="black", font=small_font)
                video.extend([np.asarray(canvas)]*12)
    fig.text(.015, .035, "Mask: 0 = later warp, 1 = earlier warp. Error colors share the fixed range [0, 0.12].", fontsize=11)
    fig.savefig(ART / "mask_examples.png", dpi=150)
    plt.close(fig)
    np.savez_compressed(ART / "demo_triplets.npz", frames=np.stack(demo), keys=np.array([x[1] for x in chosen]))
    (ART / "example_selection.json").write_text(json.dumps([dict(label=label, key=key, delta_psnr=float(gain[key]),
                                                               interior_delta_psnr=float(interior_gain[key])) for label, key in chosen], indent=2))
    imageio.mimsave(ART / "interpolation_demo.mp4", video, fps=12, codec="libx264", quality=8, macro_block_size=1)
    print("Saved visual examples, small demo bundle, and MP4", flush=True)


if __name__ == "__main__":
    main()
