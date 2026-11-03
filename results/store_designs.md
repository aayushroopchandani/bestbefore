# Store pack designs and the v1 split

Pack design ids for every photo of store sets v1 and v2, named by looking
at the photos (drafted by Claude, approved by the owner), and store set v1
split by design into training and validation
([plan](../docs/m2b-store-and-size.md), "Design ids" and "The split").
The ids and the split stay in `data/store/`: this file holds counts and
hashes only, never a photo or a design's name.

- **Designs:** 198 photos, frozen 2026-11-03
- **SHA-256 of `data/store/designs.json`:** `fa888edb77c14123c5839044e7ac880aa8de2fc299ca0f8acdfd7376bc566f6e`
- **Typed against:** v1's labels `7b9a4043…`, v2's `f1fe6b8d…`
- **The split:** `store_train.split_v1`, seed 0, SHA-256 `4418c65be61a57e1f949b7beb3013b7843eeec2a349f70496bcce6a7d2af02ca`

## Designs

|  | designs | photos | expiries in frame |
|---|---:|---:|---:|
| v1 | 40 | 99 | 81 |
| v2 | 54 | 99 | 68 |
| v2, design in v1-train (seen) | 6 | 16 | 4 |
| v2, design in v1-val only | 3 | 14 | 7 |
| v2, design not in v1 (unseen) | 45 | 69 | 57 |
| designs in both, their v1 and v2 photos | 9 | 55 | 27 |

Expiries in frame are expiry boxes. Seen means trained on: a v2 photo
whose design has photos in v1-train.

|  | photos per design (median, most) | designs with one photo | expiries: largest design, largest 4 | designs with an expiry in frame |
|---|---:|---:|---:|---:|
| v1 | 1, 11 | 23 | 10, 31 of 81 | 34 |
| v2 | 1, 8 | 30 | 5, 15 of 68 | 46 |

## The v1 split

Each design takes its photos' commonest tag (`other-face` only if every
photo is); within each tag, 30% of the designs go to validation.

| design tag | designs | train | val |
|---|---:|---:|---:|
| other-face | 4 | 3 | 1 |
| sideways | 5 | 3 | 2 |
| curved | 0 | 0 | 0 |
| dot-matrix | 9 | 6 | 3 |
| small-print | 6 | 4 | 2 |
| clean | 16 | 11 | 5 |
| **all** | 40 | 27 | 13 |

By each photo's own tag, the way results are reported:

| photo tag | photos train | expiries train | dates train | photos val | expiries val | dates val |
|---|---:|---:|---:|---:|---:|---:|
| other-face | 8 | 0 | 5 | 4 | 0 | 2 |
| sideways | 11 | 10 | 29 | 5 | 5 | 10 |
| curved | 0 | 0 | 0 | 0 | 0 | 0 |
| dot-matrix | 18 | 17 | 39 | 3 | 3 | 7 |
| small-print | 10 | 8 | 24 | 7 | 7 | 14 |
| clean | 23 | 21 | 49 | 10 | 10 | 20 |
| **all** | 70 | 56 | 146 | 29 | 25 | 53 |

Dates are every labelled date box (expiry, mfg and other).

Validation holds 25 expiries in frame, at least the 20 S4's check asks for.
