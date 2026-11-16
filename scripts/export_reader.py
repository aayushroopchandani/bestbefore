"""M3: export a CTC run's kept checkpoint to ONNX, and check it reads as
PyTorch did (docs/m3-date-reader.md, decision 16).

    python scripts/export_reader.py --run results/runs/<time>-m3-crnn-all-s0 --threads 8

train_reader.py does this at the end of every run; this script does it
again for a run on its own (a new export, a parity check on another
machine). The ONNX file goes beside the checkpoint (reader.onnx), the
result into the run's run.json under `export`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from bestbefore import runs
from bestbefore.reader import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="the run's folder under results/runs/")
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument("--suite", default="real-val")
    args = parser.parse_args()
    record = json.loads((args.run / "run.json").read_text())
    checkpoint = Path(record["checkpoints"]) / "best.pt"
    exported = train.export_checked(checkpoint, Path(record["config"]["root"]), args.threads, args.suite)
    runs.update_run(args.run, export=exported)
    print(json.dumps(exported, indent=2))
    sys.exit(0 if exported["passed"] else 1)


if __name__ == "__main__":
    main()
