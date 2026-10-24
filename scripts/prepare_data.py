"""Download Vimeo triplets and materialize the published train/val/test ID lists."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import requests

TRAIN = "bijinc/vimeo-90k-medium"
TEST = "paschalc24/vimeo-90k-test-full"


def download(job):
    repo, name, root = job
    path = root / repo.replace("/", "__") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{name}"
    tmp = path.with_suffix(path.suffix + ".partial")
    # Restart a partial HTTP transfer; complete files are reused unchanged.
    with requests.get(url, stream=True, timeout=(30, 180)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        start = last = time.monotonic()
        done = 0
        with tmp.open("wb") as f:
            for block in r.iter_content(4 * 1024**2):
                f.write(block)
                done += len(block)
                now = time.monotonic()
                if now - last > 10:
                    print(f"download {name}: {done/1e6:.0f}/{total/1e6:.0f} MB, {done/(now-start)/1e6:.1f} MB/s", flush=True)
                    last = now
    tmp.replace(path)
    print(f"download complete {repo}/{name}: {path.stat().st_size/1e6:.1f} MB", flush=True)
    return path


def records(paths):
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                frames = []
                for col in ("im1", "im2", "im3"):
                    frame = np.asarray(Image.open(io.BytesIO(row[col]["bytes"])).convert("RGB"))
                    assert frame.shape == (256, 448, 3), (row["sequence_id"], frame.shape)
                    frames.append(frame)
                yield str(row["sequence_id"]), np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(__file__).resolve().parents[1] / "data"
    selected = {s: json.loads((manifest_dir / f"{s}_manifest.json").read_text())
                for s in ("train", "val", "test")}
    jobs = [(TRAIN, f"data/train-{i:05d}-of-00025.parquet", root / "downloads") for i in range(3)]
    jobs += [(TEST, f"data/test-{i:05d}-of-00004.parquet", root / "downloads") for i in range(4)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(download, jobs))
    for splits, source_paths in [(('train', 'val'), paths[:3]), (('test',), paths[3:])]:
        output = {s: np.lib.format.open_memmap(root / f"{s}_frames.npy", mode="w+", dtype=np.uint8,
                  shape=(len(selected[s]), 3, 3, 256, 448)) for s in splits}
        lookup = {row['key']: (s, i) for s in splits for i, row in enumerate(selected[s])}
        loaded = set()
        for key, frames in records(source_paths):
            if key not in lookup or key in loaded:
                continue
            split, i = lookup[key]
            output[split][i] = frames.transpose(0, 3, 1, 2)
            loaded.add(key)
            if len(loaded) % 250 == 0:
                print(f"{splits}: loaded {len(loaded)}/{len(lookup)} triplets", flush=True)
            if len(loaded) == len(lookup):
                break
        missing = set(lookup)-loaded
        if missing:
            raise ValueError(f"Dataset is missing {len(missing)} requested sequence IDs")
        for arr in output.values():
            arr.flush()
    for split, rows in selected.items():
        (root / f"{split}_manifest.json").write_text(json.dumps(rows, indent=2))
    report = dict(train_source=f"https://huggingface.co/datasets/{TRAIN}",
                  test_source=f"https://huggingface.co/datasets/{TEST}",
                  counts={s: len(v) for s, v in selected.items()},
                  split_rule="Fixed sequence IDs from the published train/val/test lists",
                  limitation="Directory prefixes are not verified original-video IDs; similar scenes may remain.",
                  frame_order=["im1", "im2 (ground truth)", "im3"])
    (root / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
