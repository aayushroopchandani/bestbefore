"""Run Scanify's reader on a list of photos: baseline B's other half.

    $SCANIFY_DIR/.venv/bin/python scripts/scanify_runner.py $SCANIFY_DIR paths.txt

Started by bestbefore.baselines.scanify, not by hand. It runs in Scanify's
own venv and imports nothing from this project, so B is exactly the setup
Scanify was measured with, and no Scanify code is copied here (decision 8).

paths.txt holds one photo path per line. Writes one JSON line per photo to
stdout, in the same order: the expiry as `scan()` returns it (an ISO date,
a month-only pack given its month's last day) plus the OCR lines.
"""
import json
import sys
from pathlib import Path


def main() -> None:
    scanify_dir, path_list = Path(sys.argv[1]), Path(sys.argv[2])
    sys.path.insert(0, str(scanify_dir / "backend"))
    from app import ocr
    from app.scanner import scan

    print(f"scanify: OCR warmed up in {ocr.warmup()} s", file=sys.stderr)
    for path in path_list.read_text().splitlines():
        try:
            result = scan(path)
            row = {"expiry": result["expiry"], "lines": result["ocr"]["lines"]}
        except Exception as e:      # one bad photo mustn't end the run
            row = {"expiry": None, "lines": [], "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
