"""Check data separation, original-file preservation and completed deliverables."""
import hashlib
import itertools
import json
from pathlib import Path
import nbformat
from pptx import Presentation

ROOT = Path(__file__).resolve().parents[1]
manifests = {s: json.loads((ROOT / f"data/{s}_manifest.json").read_text()) for s in ("train", "val", "test")}
assert {s: len(r) for s, r in manifests.items()} == dict(train=1200, val=150, test=3782)
for a, b in itertools.combinations(manifests, 2):
    assert not {r["key"] for r in manifests[a]} & {r["key"] for r in manifests[b]}
    assert not {h for r in manifests[a] for h in r["frame_sha256"]} & {h for r in manifests[b] for h in r["frame_sha256"]}
for name, expected in json.loads((ROOT / "originals/sha256.json").read_text()).items():
    assert hashlib.sha256((ROOT / "originals" / name).read_bytes()).hexdigest() == expected
    original = ROOT.parent / name
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
nb = nbformat.read(ROOT / "occlusion_aware_frame_interpolation_final.ipynb", as_version=4)
cells = [c for c in nb.cells if c.cell_type == "code"]
assert [c.execution_count for c in cells] == list(range(1, len(cells)+1))
assert not any(o.output_type == "error" for c in cells for o in c.outputs)
assert len(Presentation(ROOT / "presentation/occlusion_aware_frame_interpolation.pptx").slides) == 10
assert (ROOT / "presentation/occlusion_aware_frame_interpolation.pdf").read_bytes().startswith(b"%PDF")
paths = [ROOT / "occlusion_aware_frame_interpolation_final.ipynb",
         *sorted((ROOT / "presentation").glob("*.pdf")), *sorted((ROOT / "presentation").glob("*.pptx")),
         *sorted((ROOT / "runs").glob("*/best.pt")), *sorted((ROOT / "artifacts").glob("*.csv"))]
report = dict(status="passed", original_notebooks_unchanged=True, exact_frame_disjoint_splits=True,
              executed_code_cells=len(cells), slide_count=10,
              sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
(ROOT / "artifacts/delivery_verification.json").write_text(json.dumps(report, indent=2))
print("PASS: split disjointness, original hashes, executed notebook, ten-slide presentation")
