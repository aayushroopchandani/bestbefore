"""Fetch the generator's fonts into $GENERATED_DIR/fonts/ and check them
(docs/m2-synthetic-generator.md, S4).

    python scripts/fetch_fonts.py

Every file comes from the Google Fonts repository at the commit pinned in
synth/fonts.py and must match its recorded SHA-256. A file already there is
checked, not fetched again. Then every face must have a glyph for every
character the text can print (text.CHARSET, looked up in its character map
with fontTools), exactly the axes its instance names, and an SIL OFL or
Apache 2.0 licence in its name table. Exits with an error on any problem.

On the cluster, run this on the login node: compute nodes have no
internet. The fonts are never committed.

Needs fonttools: pip install -e ".[synth]"
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from bestbefore.synth import fonts
from bestbefore.synth.text import CHARSET


def fetch(file: str, dest: Path) -> None:
    """Download one file into place; a cut-off download never looks done."""
    url = f"{fonts.RAW_URL}/{fonts.COMMIT}/{urllib.parse.quote(file)}"
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(part, dest)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    root = fonts.font_dir()
    for file in fonts.SHA256:
        dest = root / file
        if not dest.exists():
            print(f"fetching {file}")
            fetch(file, dest)

    found = []
    for family in fonts.FAMILIES:
        for face in fonts.faces(family):
            found += fonts.problems(face, CHARSET)
        weights = "regular, bold" if family.bold else "regular only"
        print(f"  {family.name:17s} {family.split:9s} {family.look:10s} {family.licence:7s} {weights}")
    if found:
        sys.exit("\n".join(found))
    print(f"{len(fonts.SHA256)} files in {root}: hashes, {len(CHARSET)} characters, "
          f"axes and licences all checked")


if __name__ == "__main__":
    main()
