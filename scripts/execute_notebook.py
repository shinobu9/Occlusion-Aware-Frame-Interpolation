"""Execute the default analysis/inference path in a fresh project Python kernel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", sys.prefix,
                "--name", "vfi-project", "--display-name", "Python (VFI project)"], check=True)
os.environ["JUPYTER_PATH"] = str(Path(sys.prefix) / "share/jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
path = ROOT / "occlusion_aware_frame_interpolation_final.ipynb"
nb = nbformat.read(path, as_version=4)
start = time.monotonic()
NotebookClient(nb, timeout=600, kernel_name="vfi-project", resources={"metadata": {"path": str(ROOT)}}).execute()
cells = [c for c in nb.cells if c.cell_type == "code"]
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type == "error" for c in cells for o in c.outputs)
nbformat.validate(nb)
nbformat.write(nb, path)
(ROOT / "artifacts/notebook_execution.json").write_text(json.dumps(dict(
    code_cells=len(cells), errors=0, seconds=time.monotonic()-start,
    python=sys.version, gpu_visibility=os.environ.get("CUDA_VISIBLE_DEVICES")), indent=2))
print("Run All passed", len(cells), "code cells", flush=True)
