# Store set v3

More photos from the retailer's stores, training data only, beside store
set v1 ([plan](../docs/m2b-store-and-size.md), S5b); never a test set. The
photos and the labels never enter git. This file records what was frozen,
so any change shows: `store.read_photo_list()` and `store.read_frozen()`
refuse a file or photo that differs.

- **Photo list:** 15 photos, frozen 2026-11-04
- **SHA-256 of `data/store/v3/photos.json`:** `8b279416fe0eb857e2027bd6035db11e15ced51206aa094e92d5ec4a59accd2c`
- **Version:** v3, frozen 2026-11-04
- **Photos:** 15
- **SHA-256 of `data/store/v3/train_set.json`:** `6b60780049d3ed53e0330eaf3d6819f95436974dcccfba37dc743321012fe901`

## Counts

| tag | photos | with an expiry | to the day | to the month | derived | unreadable |
|---|---:|---:|---:|---:|---:|---:|
| other-face | 0 | 0 | 0 | 0 | 0 | 0 |
| sideways | 8 | 8 | 1 | 7 | 2 | 0 |
| curved | 0 | 0 | 0 | 0 | 0 | 0 |
| dot-matrix | 0 | 0 | 0 | 0 | 0 | 0 |
| small-print | 1 | 1 | 1 | 0 | 0 | 0 |
| clean | 6 | 6 | 5 | 1 | 1 | 0 |
| **all** | 15 | 15 | 7 | 8 | 3 | 0 |

Each photo has the first tag that applies, in the order above.
`small-print` vs `clean` comes from the expiry box: under 1.5% of the
photo's long side is small print.

## History

- **v3 photo list** (2026-11-04): 15 photos, before any labelling.
- **v3** (2026-11-04): labels frozen.
- **v3** (2026-11-04): labels frozen again: one mfg date misread when Claude drafted the fields (a lid's DOP 10/23, frozen as 10/25), found at full size and fixed before anything used the first freeze (bce9c59d…).
