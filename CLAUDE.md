# BestBefore

Trained models that read the expiry date printed on product packaging, and
tell from a camera photo whether that date will be readable at all. A
student portfolio project: trained on public data on a university HPC
cluster, tested on real store photos.

---

## Why this exists

It grows out of **Scanify**, an app the owner built for a pet-supplies
retailer. The retailer's inventory system has no expiry dates, so staff
photograph each pack and the app reads two things from the photo: the
barcode (`zxing-cpp`) and the printed expiry date (`RapidOCR`, then a
hand-written rule-based date parser).

What Scanify measured on its store photos:

- **32 of 44** photos with a known expiry are read right, 12 missed, 0 read
  wrong.
- The parser extracts every date the OCR returns. The limit is **reading**
  the print, not parsing the text.
- The misses: dot-matrix print on curved caps and jar bottoms, sideways
  stamped dates, small print, and some large, clear "Best Before" lines the
  OCR skipped entirely.
- A blur score (variance of the Laplacian) did **not** predict failures.
  Photos that were read and photos that were missed scored the same. The
  score mostly measured how much fine print was in the frame, and the one
  photo known to be out of focus was read.

This project replaces hand-written rules with trained models, starting
where the rules ran out.

## Goal

Two models, each beating a measured baseline on real photos held out from
training:

1. **The date reader.** Find every date on a pack, tell the expiry date from
   the manufacture date, and read it as day / month / year.
2. **The readability checker.** Given a photo, predict whether the date
   reader will read it right. This is the useful version of a blur score:
   it learns what actually makes reading fail, not just focus. It must be
   small enough to run in a phone browser on the live camera preview, so
   staff are warned before they take the photo.

Success is a results table **broken down by failure type** (dot-matrix,
sideways, curved, small print, clean), not one headline number.

---

## Data

### Public: for training

**ExpDate** (Seker & Ahn, 2022). Licence CC BY 4.0: cite the paper and
credit KIST (Korea Institute of Science and Technology). Official page,
with a Google Drive download:
https://felizang.github.io/expdate/index_expdate.html. Annotations are JSON.

| Subset | Size | Used for | Milestone |
|---|---|---|---|
| Products-Real | 1,767 real photos (1,102 train / 665 test) | Detecting dates. Classes `date`, `due mark`, `production mark`, `code mark` | M1 |
| Products-Synth | ~12,000 synthetic | Detection, training only | M1 |
| Date-Synth / Date-Real | 128,000 synthetic + a real test set | Finding day / month / year inside a date crop | M3 |
| Components-Synth / -Real | 450,000 synthetic + a real test set | Reading the characters | M3 |

Citation: Seker, A. C. & Ahn, S. C. (2022). *A generalized framework for
recognition of expiration dates on product packages using fully
convolutional networks.* Expert Systems with Applications, 203, 117310.

### Public: for testing the checker only (M4)

- **SmartDoc-QA**: 4,260 smartphone photos of paper documents with
  defocus, motion blur and bad lighting, each labelled with its OCR
  accuracy. Its authors say it is not for training OCR, so use it only to
  test the checker.
- **SPAQ**: 11,125 photos from 66 phones with human sharpness ratings.
  Tests whether the checker responds to focus at all.

### Private: store photos, for testing only

About 99 photos taken in the retailer's stores. **They belong to the
retailer: never commit them, never train on them, never publish them.**
Only aggregate numbers leave the machine.

- 44 have a known expiry (Scanify `backend/tests/ground_truth.py`). The
  rest (`samples1/`, `samples3/`) still need expiry labels. That is M0.
- They need **tight** boxes drawn around each date for M1. Scanify's
  `date_panel_boxes.py` boxes are deliberately loose (the whole
  batch/MFG/expiry block), so they can't score a detector.
- Each photo gets one failure-type tag: `dot-matrix`, `sideways`, `curved`,
  `small-print`, `other-face`, `clean`.

Paths come from environment variables in `.env` (gitignored):

| Variable | Points to |
|---|---|
| `EXPDATE_DIR` | the unpacked ExpDate download |
| `STORE_PHOTOS_DIR` | the folder holding `samples1/`, `samples2/`, `samples3/` |
| `SCANIFY_DIR` | the Scanify repo, for baseline B |

### Date formats on the store packs

Indian pet-product packs print the expiry in at least 9 formats:
`04/12/2026`, `April 2031`, `10/2027`, `20260518`, `2028.JAN.21`,
`MM/YYYY-MM/YYYY` ranges and more. Some print only a manufacture date plus
a shelf life ("Best Before 1 Year from Mfg. Dt."), so the expiry has to be
worked out. ExpDate may not cover every one of these formats; the M2
generator must.

---

## Baselines

- **A: RapidOCR off the shelf, plus a simple date regex.** Public and
  reproducible. Run it on ExpDate.
- **B: Scanify's reader** (`app.scanner.scan` in `$SCANIFY_DIR`), on ExpDate
  and on the store photos. Call it by path. Never copy Scanify code into
  this repo.

## Milestones

Plan each milestone as a sliced doc in `docs/` before writing its code.

| # | Milestone | Done when |
|---|---|---|
| **M0** | **Data and test set.** Download Products-Real and Products-Synth; draw the boxes on ~50 photos as a contact sheet; collect stats (box size relative to the photo, dates per photo, class counts); convert the JSON to YOLO format. Label the remaining store photos, draw tight date boxes, tag failure types, then freeze the test set. Run baselines A and B. | Baseline table on ExpDate test and on the store photos, by tag |
| **M1** | **Date detector.** Fine-tune a small YOLO on Products-Synth plus the Products-Real train split. | Recall and mAP@0.5 on Products-Real test; expiry-box recall on store photos |
| **M2** | **Synthetic generator.** Dot-matrix and inkjet fonts, the store date formats, wrapped onto cylinders, rotated sideways, with glare and small print. Retrain M1 with and without it. | An ablation showing what synthetic data adds on the store photos |
| **M3** | **Date reader.** A recogniser trained on crops (Date-Synth, Components-Synth, M2 data), or a small vision-language model fine-tuned to return `{expiry, mfg, format}` as JSON. | End-to-end date accuracy vs baselines A and B, by tag |
| **M4** | **Readability checker.** Make millions of damaged copies (defocus, motion blur, glare, low light, JPEG, rotation), label each by whether the M3 reader still reads it, and train a small classifier on that label. | AUROC on held-out copies; correlation with OCR accuracy on SmartDoc-QA and with sharpness on SPAQ; results on store photos |
| **M5** | **On the phone.** Export the checker to ONNX, quantise it, run it in a phone browser. | Size, latency per frame, accuracy kept after quantising |

## Metrics

- **Detection:** recall and mAP@0.5. On store photos, did it find the
  expiry date (IoU ≥ 0.5)?
- **Reading:** exact date match on day, month and year; on month and year
  only when the pack prints no day. Count **wrong reads separately from
  no reads.** A wrong date is worse than none: reading `08/2027` as
  `08/2021` would mark a whole batch as expired.
- **Checker:** AUROC for "will be read right". Also report the agreement
  with OCR accuracy on SmartDoc-QA and with sharpness ratings on SPAQ.
- Always report **per failure tag**. Never train on, or tune
  hyperparameters against, the store photos or any test split.

## Lessons already measured (don't re-learn them)

- **Crops hurt off-the-shelf OCR.** RapidOCR read fewer dates from perfect
  hand-drawn crops (25/44) than from the whole photo (31/44). So detector
  plus RapidOCR on crops will **not** beat baseline B. The gain needs a
  recogniser trained on crops (M3). Judge M1 only on finding dates.
- **Heavy preprocessing hurts OCR.** Adaptive thresholding made it worse;
  mild sharpening helped slightly. Models trained on natural photos want
  natural photos.
- **Resolution:** for RapidOCR, a longest side of 1000 px beat 1600, 2200
  and the original size.
- **A blur score over the whole frame measures detail, not focus.** If blur
  is scored again, score only the region around the text.
- About a third of the store packs have the date and the barcode on
  different faces. That is a capture problem, not a reading problem; tag
  it `other-face` and leave it out of reading accuracy.

---

## HPC

Checked on 2026-10-14. Login details stay in `~/.ssh/config`, never in
this repo.

- **Cluster and scheduler:** the university cluster (OpenHPC on Red Hat),
  **Slurm**.
- **GPUs:** NVIDIA **H100**, 2 per node, on 2 GPU nodes (`cn5`, `cn6`:
  48 CPUs and about 500 GB RAM each, shared with other users). Partitions
  with GPUs: `gpu` (3-day limit), `medium` (2 days), `long` (7 days).
  CPU-only: `short` (6 h), `defaultq` (default, 3 days), `smp` (7 days).
  QOS `limit_gpu_qos` allows **2 jobs at a time** per user; one H100 per
  job is plenty for every model here.
- **Storage:** home is on a shared 114 TB parallel filesystem
  (`/fs1/home`). **Quota: 950 GB** (hard limit 1 TB), no limit on the
  number of files; about 65 GB was already used on 2026-10-14. **There is
  no scratch filesystem** and `$SCRATCH` is unset; no purge policy was
  found. Millions of full-size damaged photos (M4) would not fit: make
  them on the fly, or store small crops (a million 30 KB crops is 30 GB).
- **Internet:** the login node has it; **compute nodes don't** (`curl`
  exits with code 7, "couldn't connect"). Download data, pip packages and
  model weights on the login node before submitting a job.
- **Python:** modules `Python/3.11.11` (same minor version as the laptop),
  `cuda/12.6.3`, `anaconda3/2024.10`. The system `python3` is 3.9: don't
  use it.

Conventions:

- Job scripts live in `slurm/`.
- Data and checkpoints go outside the git checkout, in
  `~/bestbefore-data/`. (The original rule said "on scratch, never in
  home", but this cluster has no scratch.)
- Synthetic generation and OCR labelling run as **job arrays**: every image
  is independent.
- Every run records its config, git commit and random seed.

## Repo layout

```
bestbefore/
├── CLAUDE.md
├── README.md            the public face: problem, results, how to reproduce
├── .env.example         EXPDATE_DIR, STORE_PHOTOS_DIR, SCANIFY_DIR
├── docs/                milestone plans (sliced), dataset notes
├── src/bestbefore/      datasets, format conversion, synthesis, models, eval
├── scripts/             command-line entry points
├── slurm/               job scripts
├── experiments/         one folder per question: script + README
├── data/                gitignored
└── results/             gitignored, except the summary tables
```

---

## Working with the owner

- The owner is a student building this for their portfolio. Explain why a
  choice is made in a sentence or two, and give a recommendation rather
  than a survey of options.
- **Plans:** simple, optimised and clean, split into slices, as markdown in
  `docs/`. Read the existing docs first. Call out any departure from a
  decision already written down. Put deferred ideas in a "Later" section;
  don't drop them.
- **Experiments:** one folder per question under `experiments/`, holding
  the script and a README with the question, method, result and decision.
  The README's numbers must be reproducible from its script.
- **Report honestly.** A model that doesn't beat the baseline is still a
  result; write it down.
- **Commits:** lowercase conventional prefixes (`feat:`, `fix:`, `test:`,
  `docs:`, `refactor:`), one logical change each, plan doc first, and every
  commit should still run on its own. **No `Co-Authored-By` trailer.** When
  the owner asks for commit text, give each commit as `git add <files>` and
  `git commit -m "..."` together in one code block.
- **Never commit:** store photos, ExpDate files, model weights, `.env`.
