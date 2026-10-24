"""Run the finite cache + three-mask stage; preserve logs and fail on any child error."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
GPUS = [1, 3, 4, 6, 7]


def check_free(gpus):
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    mapping = {line.split(",")[1].strip(): int(line.split(",")[0]) for line in query.strip().splitlines()}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = {mapping[line.split(",")[0].strip()] for line in apps.strip().splitlines() if line.strip()}
    assert not busy.intersection(gpus), f"Requested GPUs now busy: {busy.intersection(gpus)}"


def available_gpus(candidates):
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    mapping = {line.split(",")[1].strip(): (int(line.split(",")[0]), int(line.split(",")[2])) for line in query.strip().splitlines()}
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
    busy = {line.split(",")[0].strip() for line in apps.strip().splitlines() if line.strip()}
    return [idx for uuid, (idx, mem) in mapping.items() if idx in candidates and uuid not in busy and mem < 200]


def launch(gpu, name, command):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    (ROOT / "logs").mkdir(exist_ok=True)
    log = (ROOT / "logs" / f"{name}.log").open("a")
    p = subprocess.Popen([sys.executable, *command], env=env, stdout=log, stderr=subprocess.STDOUT)
    print(f"START {name} GPU={gpu} PID={p.pid}", flush=True)
    return name, p, log


def finish(jobs):
    failures = []
    for name, p, log in jobs:
        rc = p.wait()
        log.close()
        print(f"FINISH {name} rc={rc}", flush=True)
        if rc:
            failures.append((name, rc))
    if failures:
        raise RuntimeError(failures)


def main():
    check_free(GPUS)
    subprocess.run([sys.executable, "scripts/cache_flows.py", "--init"], check=True)
    finish([launch(gpu, f"cache_{rank}", ["scripts/cache_flows.py", "--rank", str(rank), "--world", str(len(GPUS))]) for rank, gpu in enumerate(GPUS)])
    # Verify all shards exist before relying on a partially filled array.
    for split in ("train", "val", "test"):
        assert all((ROOT / "data" / f"{split}_cache_rank{r}.json").exists() for r in range(len(GPUS)))
    (ROOT / "data/cache_DONE.json").write_text(json.dumps(dict(gpus=GPUS, completed=time.time())))
    check_free(GPUS)
    jobs = [launch(gpu, f"mask_s{seed}", ["scripts/train_mask.py", "--run", f"runs/mask_s{seed}", "--seed", str(seed)]) for gpu, seed in zip(GPUS, (42, 43, 44))]
    jobs.append(launch(6, "heuristic_validation", ["scripts/evaluate_masks.py", "--split", "val"]))
    finish(jobs)
    (ROOT / "runs/stage1_DONE.json").write_text(json.dumps(dict(completed=time.time())))


if __name__ == "__main__":
    main()
