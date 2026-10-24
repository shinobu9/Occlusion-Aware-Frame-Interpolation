"""Prepare a pinned Vimeo subset, check IDs and decoded-frame hashes, save manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import requests

TRAIN = ("bijinc/vimeo-90k-medium", "17ba080aa556c0050d6c560115259e26c03291f9")
TEST = ("paschalc24/vimeo-90k-test-full", "7af0ffe8b63dabcadb4dee8cf8c543e269dffb5b")


def download(job):
    repo, rev, name, root = job
    path = root / repo.replace("/", "__") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    url = f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
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


def hashes(frames):
    return [hashlib.sha256(f.tobytes()).hexdigest() for f in frames]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=1200)
    ap.add_argument("--val", type=int, default=150)
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(TRAIN[0], TRAIN[1], f"data/train-{i:05d}-of-00025.parquet", root / "downloads") for i in range(3)]
    jobs += [(TEST[0], TEST[1], f"data/test-{i:05d}-of-00004.parquet", root / "downloads") for i in range(4)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(download, jobs))
    test_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths[3:])
    assert test_rows == 3782, test_rows
    test = np.lib.format.open_memmap(root / "test_frames.npy", mode="w+", dtype=np.uint8, shape=(test_rows, 3, 3, 256, 448))
    test_ids, test_hashes, test_manifest = set(), set(), []
    for i, (key, frames) in enumerate(records(paths[3:])):
        assert key not in test_ids, key
        hs = hashes(frames)
        test_ids.add(key)
        test_hashes.update(hs)
        test[i] = frames.transpose(0, 3, 1, 2)
        test_manifest.append(dict(index=i, key=key, frame_sha256=hs))
        if (i + 1) % 250 == 0:
            print(f"test decoded and hashed {i+1}/{test_rows}", flush=True)
    test.flush()
    selected = {"train": [], "val": []}
    output = {s: np.lib.format.open_memmap(root / f"{s}_frames.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 3, 256, 448)) for s, n in (("train", args.train), ("val", args.val))}
    limits = {"train": args.train, "val": args.val}
    seen_ids, seen_hashes = set(), set()
    excluded = []
    # Prefix partitions are coarser than triplets, but are NOT asserted to be source-video IDs.
    val_prefixes = {f"{i:05d}" for i in range(1, 79) if int(hashlib.sha256(f"split42/{i}".encode()).hexdigest(), 16) % 5 == 0}
    for key, frames in records(paths[:3]):
        hs = hashes(frames)
        if key in test_ids or any(h in test_hashes for h in hs):
            excluded.append(dict(key=key, reason="test overlap"))
            continue
        if key in seen_ids or any(h in seen_hashes for h in hs):
            excluded.append(dict(key=key, reason="pool duplicate"))
            continue
        split = "val" if key.split("/")[0] in val_prefixes else "train"
        if len(selected[split]) >= limits[split]:
            continue
        idx = len(selected[split])
        output[split][idx] = frames.transpose(0, 3, 1, 2)
        selected[split].append(dict(index=idx, key=key, frame_sha256=hs))
        seen_ids.add(key)
        seen_hashes.update(hs)
        if all(len(selected[s]) == limits[s] for s in limits):
            break
    assert all(len(selected[s]) == limits[s] for s in limits), {s: len(v) for s, v in selected.items()}
    for a in output.values():
        a.flush()
    selected["test"] = test_manifest
    for split, rows in selected.items():
        (root / f"{split}_manifest.json").write_text(json.dumps(rows, indent=2))
    report = dict(train_source=TRAIN, test_source=TEST, counts={s: len(v) for s, v in selected.items()},
                  val_prefixes=sorted(val_prefixes), excluded=excluded,
                  split_rule="mirror train subset, exclude all test IDs and exact decoded-frame hashes, disjoint directory prefixes for train/val",
                  limitation="Directory prefixes are not verified original-video IDs; near-duplicate scenes are not excluded.",
                  frame_order=["im1", "im2 (ground truth)", "im3"],
                  files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (root / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("counts", "excluded", "split_rule")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
