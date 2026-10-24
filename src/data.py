"""Small immutable arrays shared by all experiments."""
import json
from pathlib import Path
import numpy as np
import torch


class CachedData:
    def __init__(self, root, split):
        root = Path(root)
        self.features = np.load(root / f"{split}_features.npy", mmap_mode="r")
        self.frames = np.load(root / f"{split}_frames.npy", mmap_mode="r")
        self.regions = np.load(root / f"{split}_regions.npy", mmap_mode="r")
        self.manifest = json.loads((root / f"{split}_manifest.json").read_text())

    def __len__(self):
        return len(self.frames)

    def batch(self, ids, device="cuda", crop=None, rng=None):
        if crop is None:
            feat = np.array(self.features[ids])
            target = np.array(self.frames[ids, 1])
            regions = np.array(self.regions[ids])
        else:
            H, W = self.features.shape[-2:]
            feat, target, regions = [], [], []
            for i in ids:
                y, x = int(rng.integers(H-crop+1)), int(rng.integers(W-crop+1))
                feat.append(self.features[i, :, y:y+crop, x:x+crop])
                target.append(self.frames[i, 1, :, y:y+crop, x:x+crop])
                regions.append(self.regions[i, :, y:y+crop, x:x+crop])
            feat, target, regions = np.stack(feat), np.stack(target), np.stack(regions)
        return (torch.from_numpy(feat).to(device).float(),
                torch.from_numpy(target).to(device).float()/255,
                torch.from_numpy(regions).to(device).float())


def wd_from_features(f):
    return dict(w0=f[:, :3], w1=f[:, 3:6], f01_t=f[:, 9:11]*20, f10_t=f[:, 11:13]*20,
                o0_t=f[:, 13:14], o1_t=f[:, 14:15], hole0=f[:, 15:16], hole1=f[:, 16:17])


def swap_features(f):
    return torch.cat((f[:, 3:6], f[:, :3], f[:, 6:9], f[:, 11:13], f[:, 9:11],
                      f[:, 14:15], f[:, 13:14], f[:, 16:17], f[:, 15:16], 1-f[:, 17:18]), 1)
