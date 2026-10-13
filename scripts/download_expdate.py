"""Download and unpack ExpDate subsets into $EXPDATE_DIR.

    python scripts/download_expdate.py products-real products-synth

Each zip is saved to $EXPDATE_DIR/downloads/, checked against the SHA-256
recorded in docs/datasets.md, and unpacked into its own folder. A zip that
is already there is not downloaded again, so a zip fetched in the browser
can be dropped into downloads/ instead. On the cluster, run this on the
login node: compute nodes usually have no internet.

Needs the download extra: pip install -e ".[download]"
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

from bestbefore.config import expdate_dir

# Google Drive file IDs, from the folder linked on the ExpDate page.
# A SHA-256 is recorded once a subset has been downloaded and checked.
SUBSETS = {
    "products-real": ("Products-Real.zip", "1ZyDjpA807OkYFN2ZKgkk9ps1s-UCELSn",
                      "31dff6e0045ab6f4683c9e20b5dd04fedd751c174b681193744a013637e38da6"),
    "products-synth": ("Products-Synth.zip", "1Ewnsqjm2SqdZvAN0EhVS6PulfoE59giX",
                       "b10d82a1af1268eeb58aa3831537b577fba867c02687ae2b7fa9c278e18977ec"),
    "date-real": ("Date-Real.zip", "1pIBIF4k5J5uv4EvIOiYmKugbNgTpdM4W", None),
    "date-synth": ("Date-Synth.zip", "1zz5uwphf5sk7KuuZX2fNBzAK4Mgj6DTW", None),
    "components-real": ("Components-Real.zip", "1E_IcD8Id13kYDYtWeBmjbleeic3IDEDg", None),
    "components-synth": ("Components-Synth.zip", "1fC36oDfU8Xs2E2pF0QaGWoIcvte0ahSY", None),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(subset: str, root: Path) -> None:
    filename, file_id, expected = SUBSETS[subset]
    zip_path = root / "downloads" / filename
    if not zip_path.exists():
        import gdown
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=file_id, output=str(zip_path), quiet=False)

    actual = sha256(zip_path)
    if expected and actual != expected:
        sys.exit(f"{filename}: SHA-256 {actual} does not match {expected}")
    print(f"{filename}: sha256 {actual}" + ("" if expected else "  (not yet recorded)"))

    out = root / subset
    if out.exists():
        print(f"  {out} already unpacked")
        return
    out.mkdir(parents=True)
    # zipfile strips absolute paths and '..' from member names.
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)
    print(f"  unpacked into {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("subsets", nargs="+", choices=SUBSETS)
    args = parser.parse_args()
    root = expdate_dir()
    for subset in args.subsets:
        fetch(subset, root)


if __name__ == "__main__":
    main()
