"""Two-frame inference: ground truth is never an input."""
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from .core import MaskUNet, make_warps, make_features, blend, heuristic_mask
from .data import wd_from_features


class Interpolator:
    def __init__(self, checkpoint, device=None, heuristic_strength=.5):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = Raft_Small_Weights.C_T_V2
        self.flow = raft_small(weights=weights, progress=False).to(self.device).eval()
        self.preprocess = weights.transforms()
        self.mask = MaskUNet().to(self.device).eval()
        self.mask.load_state_dict(torch.load(Path(checkpoint), map_location=self.device, weights_only=False)["model"])
        self.heuristic_strength = heuristic_strength

    @torch.inference_mode()
    def __call__(self, earlier, later):
        a, b = earlier.to(self.device), later.to(self.device)
        if a.ndim == 3:
            a, b = a[None], b[None]
        assert a.shape == b.shape and a.shape[1] == 3
        H, W = a.shape[-2:]
        ph, pw = max(128, (H+7)//8*8)-H, max(128, (W+7)//8*8)-W
        a, b = F.pad(a, (0, pw, 0, ph), mode="replicate"), F.pad(b, (0, pw, 0, ph), mode="replicate")
        aa, bb = self.preprocess(a, b)
        f01, f10 = self.flow(aa, bb)[-1].half().float(), self.flow(bb, aa)[-1].half().float()
        wd = make_warps(a, b, f01, f10)
        # Match the stored common feature precision used during training/evaluation.
        features = make_features(wd).half().float()
        wd = wd_from_features(features)
        mask = self.mask(features)
        hm = heuristic_mask(wd, self.heuristic_strength)
        out = dict(fixed=blend(wd, .5), heuristic=blend(wd, hm), learned=blend(wd, mask),
                   learned_mask=mask, heuristic_mask=hm, warp0=wd["w0"], warp1=wd["w1"])
        return {k: v[..., :H, :W].clamp(0, 1) for k, v in out.items()}
