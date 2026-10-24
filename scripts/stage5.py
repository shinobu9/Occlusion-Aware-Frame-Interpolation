"""Build and execute final deliverables only after every test shard succeeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage1 import ROOT
from scripts.stage3 import run_queue

os.chdir(ROOT)
while not (ROOT / "runs/stage4_DONE.json").exists():
    time.sleep(30)
run_queue([("interior_diagnostic", ["scripts/evaluate_interior.py"])], gpus=[1, 3, 4, 5, 6, 7])
subprocess.run([sys.executable, "scripts/make_visuals.py"], check=True)
for script in ("report_refiners.py", "finalize_docs.py", "build_notebook.py"):
    subprocess.run([sys.executable, f"scripts/{script}"], check=True)
run_queue([("refiner_visuals", ["scripts/make_refiner_visuals.py"])], gpus=[1, 3, 4, 5, 6, 7])
run_queue([("notebook_run_all", ["scripts/execute_notebook.py"])], gpus=[1, 3, 4, 5, 6, 7])
subprocess.run([sys.executable, "scripts/build_presentation.py"], check=True)
(ROOT / "runs/stage5_DONE.json").write_text(json.dumps(dict(completed=time.time())))
print("Final deliverables generated and notebook executed", flush=True)
