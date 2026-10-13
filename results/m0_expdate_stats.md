# Box stats: real-train, real-test, synth

Made by `python scripts/box_stats.py real-train real-test synth --out results/m0_expdate_stats.md`.

| | real-train | real-test | synth |
|---|---:|---:|---:|
| photos | 1102 | 665 | 11858 |
| photo long side, median (min to max) | 1125 (534 to 2048) | 1277 (556 to 2298) | 719 (163 to 5182) |
| photos with an EXIF rotation flag | 0 | 3 | 0 |
| boxes: `code` | 541 | 229 | 5518 |
| boxes: `date` | 1244 | 68 | 16674 |
| boxes: `due` | 895 | 460 | 5397 |
| boxes: `exp` | 0 | 665 | 0 |
| boxes: `prod` | 100 | 37 | 2185 |
| photos with 0 dates | 0 (0%) | 0 (0%) | 99 (1%) |
| photos with 1 date | 962 (87%) | 601 (90%) | 6844 (58%) |
| photos with 2 dates | 138 (13%) | 62 (9%) | 4915 (41%) |
| photos with 3+ dates | 2 (0%) | 2 (0%) | 0 (0%) |
| date box area, % of photo: p10 / median / p90 | 0.29 / 0.46 / 0.97 | 0.26 / 0.43 / 0.71 | 0.32 / 1.33 / 5.54 |
| date box short side at 640 px: p10 / median | 12.1 / 16.2 px | 11.9 / 15.5 px | 13.0 / 26.7 px |
| date boxes under 8 px at 640 | 0.1% | 0.0% | 0.9% |
| date box short side at 1280 px: p10 / median | 24.1 / 32.5 px | 23.8 / 31.1 px | 26.1 / 53.3 px |
| date boxes under 8 px at 1280 | 0.0% | 0.0% | 0.0% |
| tall date boxes (height > width) | 0.0% | 0.0% | 0.5% |
