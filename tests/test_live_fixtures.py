"""M2c S2: the app's fixtures are still what Ultralytics' own functions give
(scripts/live_fixtures.py; Vitest tests the TypeScript against them). Made
again into a temporary folder and compared with the committed files, so an
Ultralytics upgrade that changes its letterbox or NMS fails here."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("ultralytics")
pytest.importorskip("torchvision")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import live_fixtures  # noqa: E402


def test_the_committed_fixtures_are_ultralytics_own(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["live_fixtures.py", "--out", str(tmp_path)])
    live_fixtures.main()
    for name in ("letterbox.json", "nms.json"):
        assert (tmp_path / name).read_text() == (live_fixtures.OUT / name).read_text(), name


def test_hashed_is_exact_and_in_0_1():
    values = [live_fixtures.hashed(i) for i in range(0, 300000, 997)]
    assert all(0 <= v < 1 for v in values)
    assert live_fixtures.hashed(1) == 2654435761 / 4294967296
    assert live_fixtures.hashed(2) == (2 * 2654435761 % 4294967296) / 4294967296
