"""Convert store set v3's HEIC photos to JPEG, once (docs/m2b-store-and-size.md, S5b).

    python scripts/store_heic.py data/store/NewExtraStoreData

Each HEIC in the folder becomes a JPEG of the same name, at quality 95, in
$STORE_PHOTOS_DIR/trainset3/, by macOS's own `sips` (no new dependency).
data/store/v3/heic.json records each HEIC's SHA-256 beside its JPEG's, so
the conversion stays traceable; from then on the JPEGs are v3's photos and
are never re-encoded. Freeze their list next:
`freeze_store_test_set.py --version v3 --photos`.

It refuses to run when $STORE_PHOTOS_DIR is a git repo that doesn't ignore
trainset3/ (it is the Scanify repo, which has a GitHub remote), when
trainset3/ already holds a photo, and when a name or a JPEG's bytes match a
v1 or v2 photo. Everything it writes names photos, so none of it is
committed.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import date
from pathlib import Path

from bestbefore import store
from bestbefore.config import store_photos_dir
from bestbefore.images import orientation, upright_size

QUALITY = 95
HEIC_PATH = store.V3.data_dir / "heic.json"


def ignored_by_git(root: Path, folder: str) -> bool:
    """True when `root` isn't a git repo, or is one that ignores `folder`."""
    inside = subprocess.run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
                            capture_output=True, text=True)
    if inside.returncode != 0:
        return True
    probe = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", f"{folder}/probe.jpg"])
    return probe.returncode == 0


def known_photos(root: Path) -> tuple[set[str], set[str]]:
    """The file names (without extension, any case) and SHA-256s of v1's and v2's photos."""
    photos = store.photos_on_disk(store.V1, root) + store.photos_on_disk(store.V2, root)
    return ({Path(p).stem.casefold() for p in photos}, {store.sha256(root / p) for p in photos})


def convert(heic: Path, jpeg: Path) -> None:
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", str(QUALITY),
                    str(heic), "--out", str(jpeg)], check=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("source", type=Path, help="the folder holding the HEIC files")
    args = parser.parse_args()

    root = store_photos_dir()
    (folder,) = store.V3.folders
    out = root / folder
    heics = sorted(p for p in args.source.iterdir() if p.suffix.lower() == ".heic")
    if not heics:
        raise SystemExit(f"no .heic files in {args.source}")
    if not ignored_by_git(root, folder):
        raise SystemExit(f"{root} is a git repo that doesn't ignore {folder}/: add it to its "
                         ".git/info/exclude first (M0 D1 did this for testset2/)")
    if out.exists() and any(p.suffix.lower() in (".jpg", ".jpeg") for p in out.iterdir()):
        raise SystemExit(f"{out} already holds photos: v3 is converted once")
    if HEIC_PATH.exists():
        raise SystemExit(f"{HEIC_PATH} already exists: v3 is converted once")
    names, hashes = known_photos(root)
    clash = [h.name for h in heics if h.stem.casefold() in names]
    if clash:
        raise SystemExit(f"{len(clash)} names already used by v1 or v2 photos: {', '.join(clash)}")

    out.mkdir(exist_ok=True)
    photos = {}
    for heic in heics:
        jpeg = out / f"{heic.stem}.jpg"
        convert(heic, jpeg)
        digest = store.sha256(jpeg)
        if digest in hashes:
            raise SystemExit(f"{jpeg.name} is byte-identical to a v1 or v2 photo")
        photos[f"{folder}/{jpeg.name}"] = {"heic": heic.name, "heic_sha256": store.sha256(heic),
                                           "sha256": digest, "upright": list(upright_size(jpeg)),
                                           "exif_orientation": orientation(jpeg)}
    tool = subprocess.run(["sips", "--version"], capture_output=True, text=True).stdout.strip()
    HEIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    HEIC_PATH.write_text(json.dumps({"made": date.today().isoformat(), "tool": f"sips {tool}",
                                     "macos": platform.mac_ver()[0], "quality": QUALITY,
                                     "photos": photos}, indent=1, sort_keys=True) + "\n")
    turned = sum(p["exif_orientation"] != 1 for p in photos.values())
    print(f"converted {len(photos)} photos into {out}; {turned} carry an EXIF turn")
    print(f"wrote {HEIC_PATH}")
    print("\nnext: python scripts/freeze_store_test_set.py --version v3 --photos")


if __name__ == "__main__":
    main()
