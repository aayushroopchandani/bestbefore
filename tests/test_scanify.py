from bestbefore.baselines import scanify
from bestbefore.samples import Expiry


def test_parse_iso():
    assert scanify.parse_iso("2027-08-31") == Expiry(2027, 8, 31)
    assert scanify.parse_iso(None) is None


def _rows(reads):
    return [{"id": photo, "read": read} for photo, read in reads.items()]


def test_reproduction_counts_scanifys_way():
    truth = {"a.jpg": Expiry(2031, 4, 30), "b.jpg": Expiry(2027, 8, 15),
             "c.jpg": Expiry(2026, 1, 9), "d.jpg": None}         # d: no expiry, not counted
    rows = _rows({"a.jpg": "2031-04-30", "b.jpg": None, "c.jpg": "2026-01-08", "d.jpg": "2025-01-01"})
    check = scanify.reproduction(rows, truth)
    assert (check["n"], check["right"], check["none"], check["wrong"]) == (3, 1, 1, 1)
    assert check["matches_scanify"] is False


def test_reproduction_matches_the_measured_numbers():
    truth = {f"p{i}.jpg": Expiry(2027, 1, 1) for i in range(44)}
    rows = _rows({p: ("2027-01-01" if i < 32 else None) for i, p in enumerate(truth)})
    assert scanify.reproduction(rows, truth)["matches_scanify"] is True
