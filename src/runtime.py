import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def save_config(args, run):
    run.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  cudnn_benchmark=True, cudnn_allow_tf32=True)
    (run / "config.json").write_text(json.dumps(config, indent=2))


def rng_state(rng):
    return dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(state, rng):
    rng.bit_generator.state = state["numpy"]
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
