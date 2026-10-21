# Box stats: store, real-test

Made by `python scripts/box_stats.py store real-test --out results/m0_store_stats.md`.

| | store | real-test |
|---|---:|---:|
| photos | 99 | 665 |
| photo long side, median (min to max) | 1600 (1599 to 4032) | 1277 (556 to 2298) |
| photos with an EXIF rotation flag | 46 | 3 |
| boxes: `code` | 0 | 229 |
| boxes: `date` | 0 | 68 |
| boxes: `due` | 0 | 460 |
| boxes: `exp` | 0 | 665 |
| boxes: `expiry` | 81 | 0 |
| boxes: `mfg` | 91 | 0 |
| boxes: `other_date` | 27 | 0 |
| boxes: `prod` | 0 | 37 |
| photos with 0 dates | 7 (7%) | 0 (0%) |
| photos with 1 date | 8 (8%) | 601 (90%) |
| photos with 2 dates | 63 (64%) | 62 (9%) |
| photos with 3+ dates | 21 (21%) | 2 (0%) |
| date box area, % of photo: p10 / median / p90 | 0.10 / 0.19 / 0.56 | 0.26 / 0.43 / 0.71 |
| date box short side at 640 px: p10 / median | 7.2 / 11.0 px | 11.9 / 15.5 px |
| date boxes under 8 px at 640 | 17.6% | 0.0% |
| date box short side at 1280 px: p10 / median | 14.4 / 22.0 px | 23.8 / 31.1 px |
| date boxes under 8 px at 1280 | 0.5% | 0.0% |
| tall date boxes (height > width) | 17.6% | 0.0% |

## Store photos by failure tag

| tag | photos | with an expiry | to the day | to the month | derived | unreadable |
|---:|---:|---:|---:|---:|---:|---:|
| other-face | 12 | 0 | 0 | 0 | 0 | 0 |
| sideways | 16 | 16 | 6 | 10 | 1 | 0 |
| curved | 0 | 0 | 0 | 0 | 0 | 0 |
| dot-matrix | 21 | 21 | 19 | 2 | 1 | 0 |
| small-print | 17 | 17 | 8 | 9 | 2 | 0 |
| clean | 33 | 33 | 18 | 15 | 2 | 0 |
| **all** | 99 | 87 | 51 | 36 | 6 | 0 |

## Store expiry formats

| expiry format, as printed | photos |
|---:|---:|
| `DD/MM/YYYY` | 27 |
| `Month YYYY` | 19 |
| `MON YYYY` | 11 |
| `DD-MM-YYYY` | 8 |
| `relative: 24 months from mfg` | 4 |
| `DD/MM/YY` | 3 |
| `MON-YYYY` | 2 |
| `relative: 1 year from mfg` | 2 |
| `MM/YYYY` | 2 |
| `MM/YYYY-MM/YYYY` | 2 |
| `DDMMYYYY` | 2 |
| `DDth Month YYYY` | 2 |
| `DD Month YYYY` | 1 |
| `YYYY.MON.DD` | 1 |
| `YYYYMMDD` | 1 |
