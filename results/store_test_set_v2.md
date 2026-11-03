# Store test set v2

New photos from the retailer's stores, the test set once store photos are
trained on ([plan](../docs/m0-data-and-test-set.md), Part D). The photos
and the labels never enter git. This file records what was frozen, so any
change shows: `store.read_photo_list()` and `store.read_frozen()` refuse a
file or photo that differs.

- **Photo list:** 99 photos, frozen 2026-10-28
- **SHA-256 of `data/store/v2/photos.json`:** `8be1f9e9cb256625259c17b88a070a9a30e204c94a85ff4ad1cd498be1d6080c`
- **Version:** v2, frozen 2026-11-03
- **Photos:** 99
- **SHA-256 of `data/store/v2/test_set.json`:** `f1fe6b8d1c953a100b2245c3a1401a82bcc6314796c814609ae05ad5ad4a74f0`

## Counts

| tag | photos | with an expiry | to the day | to the month | derived | unreadable |
|---|---:|---:|---:|---:|---:|---:|
| other-face | 24 | 0 | 0 | 0 | 0 | 0 |
| sideways | 13 | 13 | 11 | 2 | 3 | 0 |
| curved | 1 | 1 | 0 | 1 | 0 | 0 |
| dot-matrix | 29 | 29 | 27 | 2 | 1 | 0 |
| small-print | 5 | 5 | 2 | 3 | 0 | 0 |
| clean | 27 | 26 | 6 | 20 | 4 | 1 |
| **all** | 99 | 74 | 46 | 28 | 8 | 1 |

Each photo has the first tag that applies, in the order above.
`small-print` vs `clean` comes from the expiry box: under 1.5% of the
photo's long side is small print.

## History

- **v2 photo list** (2026-10-28): 99 photos, before any labelling.
- **v2** (2026-11-03): labels frozen.
