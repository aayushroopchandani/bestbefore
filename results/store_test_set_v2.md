# Store test set v2

New photos from the retailer's stores, the test set once store photos are
trained on ([plan](../docs/m0-data-and-test-set.md), Part D). The photos
and the labels never enter git. This file records what was frozen, so any
change shows: `store.read_photo_list()` and `store.read_frozen()` refuse a
file or photo that differs.

- **Photo list:** 99 photos, frozen 2026-10-28
- **SHA-256 of `data/store/v2/photos.json`:** `8be1f9e9cb256625259c17b88a070a9a30e204c94a85ff4ad1cd498be1d6080c`
- **Labels:** not frozen yet

## History

- **v2 photo list** (2026-10-28): 99 photos, before any labelling.
