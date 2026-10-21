# Store test set

The retailer's photos, labelled by hand in M0
([plan](../docs/m0-data-and-test-set.md), Part B). The photos and the
labels never enter git. This file records what was frozen, so any change
shows: `store.read_test_set()` refuses a file or photo that differs.

- **Version:** v1, frozen 2026-10-21
- **Photos:** 99
- **SHA-256 of `data/store/test_set.json`:** `7b9a4043886c9f4fcb37cba339962261acaef19e2993f7fba365a75f49b588f4`

## Counts

| tag | photos | with an expiry | to the day | to the month | derived | unreadable |
|---|---:|---:|---:|---:|---:|---:|
| other-face | 12 | 0 | 0 | 0 | 0 | 0 |
| sideways | 16 | 16 | 6 | 10 | 1 | 0 |
| curved | 0 | 0 | 0 | 0 | 0 | 0 |
| dot-matrix | 21 | 21 | 19 | 2 | 1 | 0 |
| small-print | 17 | 17 | 8 | 9 | 2 | 0 |
| clean | 33 | 33 | 18 | 15 | 2 | 0 |
| **all** | 99 | 87 | 51 | 36 | 6 | 0 |

Each photo has the first tag that applies, in the order above.
`small-print` vs `clean` comes from the expiry box: under 1.5% of the
photo's long side is small print.

## History

- **v1** (2026-10-21): first version.
