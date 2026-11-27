# M2c: Live date boxes on the phone

Planned 2026-11-22, after a discussion with the owner the same day (a live
camera page before M3 resumes; the detector run on a throttled, gated
stream, never on a frame shrunk below 1,280 px; a React app built like
Scanify's frontend, so it can move into Scanify later). Status: **S1
done** (the laptop 2026-11-22, the cluster's job 2026-11-23,
[its records](#s1-the-model-files-and-their-parity--laptop-then-a-cpu-job--submits-1-cpu-job)):
R0 passes, R1 fails (the app pads square), and R2's accuracy half keeps
float32 on WebAssembly. **S2 done** (2026-11-24,
[its record](#s2-the-app-camera-worker-boxes-benches--laptop-then-the-phone)):
the app matches Ultralytics in the laptop's Chrome and on an Android
phone (WebGPU float32, 4 to 5 detections a second live); on an iPhone
WebGPU crashes the tab and WebAssembly runs about 1 a second. The target
phone is now Android, the owner's OnePlus 13 for now (Decision 15).
**S3 done** (2026-11-26,
[its record](#s3-the-frame-loop--laptop-then-the-phone)): on the Android
phone the loop runs 4 detections a second while still, shows boxes 0.47 s
after the hand stops, and rests at 1 a second on a still pack (Decision
16); the gate's copy is drawn on the GPU and read every other frame
(Decision 17), T 4.42. Next: S4, on home days. **Since 2026-11-23 M2c
and M3 alternate** (Decision 1): campus days, when the cluster is
reachable, go to M3's cluster slices (from S6, `docs/m3-date-reader.md`),
home days to M2c's laptop and phone slices; M3's two GPU jobs stay held
until its S9. The learnings guide (`learnings/m2c.md`) is written
separately, later, as for M2 and M3.

**Goal.** A React web app that opens the phone's camera and draws M2b's
date boxes on the live picture, a few times a second, with the detector
running on the phone itself. It is a demo and a dry run of M5: the export
to ONNX, quantisation, ONNX Runtime Web on WebGPU and WebAssembly, and a
camera loop in a phone browser are built and measured now, on the
detector, so that M4's checker can later drop into the same loop. It stays
private: the laptop serves it to the owner's phone, and it is never
published.

**Done when.** `results/m2c_live.md` reports, from the run records alone:

- **the exported model against PyTorch**: the M2b detector as ONNX at
  float32, float16 and int8, padded square and tight, on the five real
  validation sets (rules R0 to R2);
- **the app against Ultralytics** on the same photos: the browser's
  letterbox, model and NMS, on the laptop and on the phone;
- **speed on the owner's phone** (and in the laptop's browser): ms a
  detection by stage for each backend, precision and optimisation step
  tried (R2 to R4); the sizes of the model and the runtime; load and
  warm-up times;
- **the live loop**: detections a second, how often the stillness gate
  skips a frame, the time from holding still to boxes, and 5 minutes held
  on one pack, for heat;

and the app runs on the owner's phone from the laptop, its detector code
a self-contained module that Scanify can take later.

M2c judges speed and faithfulness, not finding dates: the detector is
M2b's, unchanged, and M2b measured its accuracy. No store set v2 photo is
touched.

---

## Why now, and why small

- **M5 needs exactly this stack**, for the checker. Finding the problems
  now (what the phone's browser supports, WebGPU against WebAssembly,
  heat) costs less than finding them at the end, and the loop built here
  is the shell M4's checker drops into.
- **A demo shows the project in seconds**: point a phone at a pack and
  boxes appear.
- **It needs no GPU**: the laptop, one short CPU job and the owner's
  phone. M3's held GPU jobs lose nothing by waiting.
- **Time box: three sessions.** Whatever isn't done after the third goes
  to [Later](#later), and M3 resumes at S6.

---

## What earlier milestones tell us

### The detector

- **The M2b detector** is `n1280-ft-s0` (yolo11n at 1,280 px, fine-tuned
  on store set v1) at `threshold_store` 0.1641 (`final_test.DETECTOR`; its
  weights' SHA-256 `c839af8a…` in `m2b.py`). Scored once on store set v2:
  65 of 68 expiries at precision 0.761. On v1-val at about that threshold
  (the laptop's own F2 threshold 0.1665, M2b S13): 60 of 63 dates, 29 of
  29 expiries, precision 0.779, 0.50 false boxes a photo; AP@0.5 for dates
  0.9596. On real-val: 0.9949, 187 of 187.
- **Four classes**: `date`, `due`, `prod`, `code` (`yolo.CLASSES`): the
  dates and the keyword marks beside them (an expiry word, a manufacture
  word, a batch code). The app draws all four, dates strongest.
- **About one false box every two photos** at that threshold. Live, a
  false box can come and go; the tracker keeps boxes from flickering but
  hides nothing. The threshold isn't changed: it is M2b's operating point,
  chosen on store photos.
- **Square padding everywhere so far.** Every number since M1 pads each
  photo to a 1,280 × 1,280 square (`rect=False` in `detector.predict`),
  because Ultralytics' tight padding moved one validation date's
  confidence by 0.15 (M1 D3; `docs/learnings/m1.md` §3.4). A 16:9 camera
  frame held upright is 720 × 1,280 at the model's size: square padding
  adds 560 grey columns, 78% more pixels than the frame. Tight padding (to
  a multiple of 32, 736 × 1,280) does 42% less work than square, but it
  departs from M1 D3, so R1 measures it before the app uses it.
- **Speed so far**: 175 ms a photo on the laptop's CPU in PyTorch (median,
  square, M2b S13); 54 ms a pass on 24 CPUs of `cn4` (M2b S13). Never
  measured in a browser or on a phone.

### Lessons that shape the app

- **A whole-frame blur score measures detail, not focus** (Scanify, M0).
  The stillness gate below measures how much the picture moves from one
  frame to the next, which is a different thing, and it only decides when
  to run the detector. It is never a readability signal: that is M4's
  checker's job.
- **Heavy preprocessing hurts** models trained on natural photos (M0). The
  app feeds the frame as it is, resized only: no sharpening, no contrast.
- **Small print is what the detector worked hardest for** (M2b, Cause 4:
  the store's dates have a median box of 1.73% of the long side, 37% under
  1.5%). So the model always sees 1,280 px on the long side, never fewer.
- **The laptop overheats on long CPU work** (CLAUDE.md). Scoring the
  export over the validation sets is a `short` job; the laptop does
  v1-val's 34 photos, and browser benches on its GPU.

---

## How the app reads the camera

```
camera stream (getUserMedia, about 30 fps, 1920 × 1080 asked)
  │ every frame (requestVideoFrameCallback)
  ▼
stillness gate: a 64 px grey copy, minus its mean, against the last copy
  │  moving: boxes dim, "hold still"; moving for 400 ms: boxes cleared
  │  still for 150 ms, no detection running, 200 ms since the last start
  ▼
createImageBitmap(video) ── transferred ──► detector worker
                                              letterbox: long side 1,280 (tight or square, R1)
                                              ONNX Runtime Web: WebGPU or WebAssembly (R2, R3)
                                              best class ≥ 0.1641, NMS at IoU 0.7 per class
  ◄── boxes in frame coordinates, each stage's time ──┘
tracker: match by IoU, smooth, keep through one miss, drop at two
  ▼
overlay canvas over the video (requestAnimationFrame)
```

**Three sizes, kept apart:**

1. **The camera's frame.** 1920 × 1080 is asked for (`ideal`), and
   whatever arrives is recorded. A frame whose long side is under 1,280 px
   gets a warning on screen: the model would see print smaller than it
   learned. 1280 × 720 is fine: its long side is exactly the model's. (This
   corrects the planning discussion, which said a 720p phone should be
   warned.)
2. **The model's input.** The frame is scaled so its long side is 1,280
   (720 × 1,280 upright), then padded to 736 × 1,280 (tight) or
   1,280 × 1,280 (square), by R1. It is never shrunk further.
3. **The gate's copy.** 64 px on the long side. It never reaches the
   model.

### The frame loop

- **One detection at a time, on the newest frame.** When a detection
  ends, the next starts from the frame on screen then; frames in between
  are dropped, never queued. A fast phone gets several detections a
  second, a slow one fewer, and boxes never fall behind a backlog.
- **The stillness gate** (the "debouncing"). For each frame, the video is
  drawn into a 64 px copy; its grey values have their mean subtracted (so
  the camera's auto-exposure changing the brightness isn't motion), and
  the mean absolute difference from the last copy is the motion value.
  Below the threshold `T` for 150 ms is *still*, and a detection starts
  only when still. Moving: boxes dim. Moving for 400 ms: boxes cleared,
  since they no longer sit on the picture. `T` comes from the owner's own
  frames by rule G, before any live measure. (S3: the copy is drawn on a
  canvas on the GPU and read every other frame on the phone, Decision 17.)
- **Rest when nothing changes.** At least 200 ms between detection starts
  (5 a second at most), and after 3 detections in a row while still, one a
  second until the picture moves. A phone held on one pack does no more
  work than it needs: less heat, less battery. (S3, Decision 16: first
  "3 with the same boxes", which the boxes' flicker never let happen.)
- **The tracker.** Each new box is matched to a shown box of its class by
  IoU (greedy, highest first, at least 0.3). A matched box moves 60% of
  the way to the new one, which smooths jitter; an unmatched new box
  appears at once; a shown box missed once stays, missed twice goes. Dates
  are drawn solid, keyword marks thin and labelled.
- **Hidden page, screen on.** The loop stops while the page is hidden; a
  screen wake lock keeps the screen on while scanning.
- **The detector runs in a Web Worker**, so the preview stays smooth on
  the main thread. ONNX Runtime Web's own proxy worker stays off: it would
  be a worker inside a worker.

### One detection

In the worker: the frame is drawn into an OffscreenCanvas at the model's
size (grey 114 around it, as Ultralytics pads); its pixels become the
float input (RGB, 0 to 1, channels first); the model runs; each anchor
keeps its best class if at the threshold; NMS runs per class at IoU 0.7
(300 boxes at most); boxes are mapped back to the frame. Each step mirrors
Ultralytics' own `LetterBox`, `non_max_suppression` and `scale_boxes`, and
the tests check them against fixtures those functions make (S2). The
browser resizes with its own filter, not OpenCV's `INTER_LINEAR`: the
benches measure whether that moves any box.

### Where it runs

- **WebGPU** where the browser has it (Chrome on recent Android phones;
  Safari from iOS 26): the model on the phone's GPU, in float16 if the GPU
  offers `shader-f16`.
- **WebAssembly** everywhere else, on the CPU: SIMD and threads (threads
  need the page to be cross-origin isolated: COOP and COEP headers), int8
  if R2 keeps it.

The app picks by R3, and shows on screen which it uses.

---

## The model files

Exported on the laptop once, hashed, and copied to the cluster, so the
job and the app run the same files:

| File | Made by | Used by |
|---|---|---|
| `det-fp32.onnx` | Ultralytics' export: height and width dynamic, batch 1, simplified by `onnxslim` | Python's scoring; WebGPU and WebAssembly, as the float32 baseline |
| `det-fp16.onnx` | `onnxruntime.transformers.float16`, inputs and outputs kept float32 | WebGPU with `shader-f16` |
| `det-fp16-head32.onnx` | the same, the Detect head's decoding kept float32: R2's one fallback | WebGPU, only if R2 says |
| `det-int8.onnx` | `onnxruntime.quantization.quantize_static`: QDQ, int8 weights per channel, uint8 activations, MinMax calibration on 200 training photos, the Detect head's decoding left in float | WebAssembly |
| `det-int8-pct.onnx` | the same with Percentile calibration (99.999): R2's one fallback | WebAssembly, only if R2 says |
| `manifest.json` | the export and the job | each file's SHA-256, size, opset and node count; ONNX Runtime's and Ultralytics' versions; the source weights' hash; the calibration list's hash |

- **Why the head stays in float for int8.** The model's output joins the
  box coordinates (0 to 1,280 px) and the class scores (0 to 1) in one
  tensor. One int8 scale for both is about 5 px a step, which rounds every
  score to 0. The head's DFL, sigmoid and that join stay float; the export
  lists the nodes it leaves out.
- **Calibration photos come from training only**: the store export's 80
  training photos, upright, and 120 of ExpDate's real training photos (a
  fixed seed), letterboxed tight as they are scored. No validation photo,
  so the check isn't flattered. Calibration runs the model with every
  intermediate tensor as an output: too heavy for the 8 GB laptop, so it
  runs in the CPU job, after an 8-photo smoke on the laptop.
- **Dynamic height and width.** One file serves upright and sideways
  frames and photos of any shape. The app pins the shape when it builds a
  session (`freeDimensionOverrides`), so ONNX Runtime still optimises for
  fixed shapes. If pinning doesn't give WebGPU a static graph (S4 checks:
  graph capture needs one), the export also writes fixed-shape files
  (`--shape 736x1280`).
- **`YOLO_AUTOINSTALL=false`** during the export: otherwise Ultralytics
  installs missing packages by itself (it wants `onnxslim>=0.1.82` to
  simplify). Here a missing package fails loudly. `onnxslim==0.1.99`
  (released 2026-11-04) joins a new `live` extra.

S4's steps (O3, O4) make further variants only if R4 keeps them.

---

## The app

A React app in `web/`, built like Scanify's frontend so that it can move
there, and written here from nothing (CLAUDE.md: no Scanify code is copied
in). The same libraries at the versions Scanify's lockfile resolves on
2026-11-22: React 19.2.8, TypeScript 6.0.3, Vite 8.2.2 with
`@vitejs/plugin-react` 6.1.1 and `@vitejs/plugin-basic-ssl` 2.3.0,
Tailwind 4.3.3, Vitest 5.0.3, oxlint 1.82.0; plus `onnxruntime-web`
1.31.1. Exact pins, `package-lock.json` committed.

```
web/
  src/detector/   the part Scanify can take: no React, no app state
    letterbox.ts      frame to model input, and boxes back to the frame
    postprocess.ts    best class, the threshold, NMS
    session.ts        backend, precision, pinned shape, warm-up
    detector.worker.ts, client.ts (the main thread's typed handle), types.ts
  src/live/       the loop without React: stillness.ts (the gate), scheduler.ts
                    (scheduling, rest, FrameLoop), tracker.ts, liveRecord.ts, stats.ts,
                    gateBench.ts, copiers.ts (S3's gate bench)
                  its hooks and the overlay: useCamera.ts, useStillness.ts,
                    useDetections.ts, usePage.ts, LiveBoxes.tsx
  src/pages/      Live, Bench (its runs in benchRun.ts), Motion (with GateBench.tsx): the
                    demo's own screens (Tailwind)
  src/api.ts      the key and the API's calls
  src/records.ts  run records, posted to the API
```

- **Live**: the loop, with one line of numbers on screen: backend ·
  precision · input shape · ms a detection · detections a second · still
  or moving.
- **Bench**, two kinds. *Parity*: photos served by the laptop, each
  letterboxed to its own shape as Ultralytics does (a session without a
  pinned shape), at conf 0.001; boxes and times posted back, for scoring.
  *Speed*: the session pinned at the camera's shape, as live, fed the same
  photos fitted into that shape; 5 warm-ups, then each stage timed. Speed
  doesn't depend on what's in the picture, so photos stand in for frames,
  and every configuration sees the same inputs.
- **Motion**: the gate's values for rule G: 20 s held still on a pack,
  20 s moving between packs.
- **Camera**: the back camera (`facingMode: environment`), 1920 × 1080 and
  30 fps asked; a menu lists the cameras, because some phones open the
  ultra-wide one (smaller print); continuous focus where the browser
  offers it (Chrome on Android; Safari focuses by itself). The camera's
  name, size and settings go into every live record.
- **The video is shown whole** (`object-fit: contain`), so every box the
  model finds is on screen; boxes map from frame to screen by the same
  scale and offset.

### Serving it to the phone

As Scanify does: Vite's dev server on the Wi-Fi (`host: true`) with a
self-signed HTTPS certificate (`plugin-basic-ssl`), because a browser
gives the camera only to a secure page. The phone warns once and the
owner accepts. **It is the same for Android and iPhone**: no cable, no
setting changed on the phone. Fallbacks, only if a browser refuses the
camera or the worker on such a page: Android Chrome over USB as
`localhost` (`adb reverse`); an iPhone trusting a local certificate
authority made with `openssl`, removed after M2c.

- **The API, in Python** (`scripts/live_api.py`, standard library, on
  `127.0.0.1` only): the model files, bench photos by set and index, and
  the run records posted back, written to `results/runs/m2c/` with the
  server's git commit, the models' manifest and the time. Vite proxies
  `/api` to it, so the page is one origin (no mixed content, no CORS), as
  in Scanify.
- **A key.** The API answers only requests that carry the key it prints
  at start (header `X-Live-Key`). The phone opens
  `https://<laptop>:5180/#k=<key>`: after the `#`, the key is never part
  of a request line or a log; the app keeps it in session storage and
  sends it as the header. Nobody else on the Wi-Fi can fetch the model,
  which was trained on the retailer's photos.
- **Store photos never reach the phone.** The API serves v1-val only when
  started with `--with-store`, for benches in the laptop's own browser;
  the phone's benches use real-val (ExpDate, public).
- **Headers**: COOP `same-origin` and COEP `require-corp` (WebAssembly
  threads need them); `application/wasm` for `.wasm`; the model files
  cached by their hash. Ports 5180 and 8765, fixed (`strictPort`), so
  Scanify's dev servers (5173, 8000) can run at the same time.
- **Measurements run on the built app** (`vite build`, then `vite preview`
  with the same proxy, HTTPS and headers), never on the dev server's
  unbundled code.

---

## Records and scores

Every record holds its config, the git commit and the models' hashes
(CLAUDE.md). Nothing in M2c is random, so there is no seed.

| Record | Made by | Holds |
|---|---|---|
| `parity-<set>-<variant>.json` | the CPU job (S1), the laptop's v1-val smoke | per photo, every box at conf ≥ 0.001; per set, AP@0.5 for dates (`detection_eval.average_precision`), dates found at IoU 0.5 at 0.1641, expiries found (v1-val), false boxes a photo; ms a pass |
| `bench-parity-*.json` | the app (S2, S4) | device (browser, CPU threads, memory, GPU adapter), backend, precision, threads, flags; model file and hash; runtime version; per photo its boxes at conf ≥ 0.001 |
| `bench-speed-*.json` | the app (S2, S4) | the same device and config, the pinned shape; per run each stage's ms; the first (cold) run, the session build and the model download apart |
| `live-*.json` | the app (S3) | camera settings, the gate's config and the loop's constants; every frame the gate read (value, cost, decision, motion), the frames skipped and missed; per detection each stage's ms; detections a second while still, awake and resting; still-to-boxes times; a median a minute |
| `motion-*.json` | the app (S3) | the gate's values, still and moving, against the frame before and the one before that; its cost a frame |
| `gate-bench-*.json` | the app's Motion page (S3) | ways of making the gate's copy: each read's ms held and ms until ready, its motion value, the frames each saw or missed; a summary per copy |

`scripts/score_live.py` scores parity benches with M2b's own code
(`detection_eval`) against the job's ONNX float32 records for the same
photos: AP, dates found at 0.1641, and box by box (each box at 0.1641
matched at IoU ≥ 0.9, and how far its confidence moved). Photos are named
by set and index only.

---

## Decision rules

Written before any number. "Found" is a date found at IoU 0.5 at 0.1641
(M2b's `threshold_store`, never chosen again); AP is ours, for dates. The
real sets are M2b's: v1-val (34 photos, 63 dates), real-val (165, 187),
real-val-rot (165, 187), real-val-small (41, 186) and real-val-rot180
(165). The synthetic sets judge training data, not a deployed model, and
are left out.

- **R0, the export (a check, not a choice).** ONNX float32 padded square
  against PyTorch on the five real sets: every date one finds, the other
  finds, and AP within 0.002 on each set. Otherwise stop: the export is
  wrong, and nothing after it means anything.
- **R1, tight padding.** Adopted if, on each real set, its AP is no more
  than 0.01 under square's and it finds at most 1 date fewer, and on
  v1-val it finds every expiry square finds. Otherwise the app pads
  square, and the report gives the cost.
- **R2, precision, per backend, at R1's padding.** int8 is used on
  WebAssembly, and float16 on WebGPU, if on each set it is scored on its
  AP is no more than 0.01 under float32's and it finds at most 1 date
  fewer, **and** it is at least 10% faster than float32 on the owner's
  phone (the median ms a detection in the speed bench). int8 is scored on
  the five real sets by the CPU job; float16 in browsers on WebGPU, v1-val
  on the laptop and real-val on the phone, each beside float32 in the same
  browser. If one fails on accuracy, its fallback (`int8-pct`,
  `fp16-head32`), scored alongside, is read the same way, once; if that
  fails too, float32.
- **R3, backend.** The app defaults to the backend with the lower median
  ms a detection on the owner's phone, each at its R2 precision; within
  10% of each other, WebGPU, which leaves the CPU to the page.
- **R4, each optimisation step (S4, in order).** Kept if it cuts the
  median ms a detection on the owner's phone by at least 10% and its boxes
  equal the step before's on the same speed-bench inputs (every box at
  0.1641 matched at IoU ≥ 0.95).
- **G, the gate's threshold.** `T` is the geometric mean of the 95th
  percentile of the still values and the 5th percentile of the moving
  values, from one Motion record, if the first is below the second. If
  they overlap, the gate is off: detection runs continuously, and the
  report says so.

---

## The optimisation ladder

Cheapest first. Each step after O2 is measured on the phone's speed bench
against the step before, and R4 keeps it or not.

| Step | What | Why it might help | For |
|---|---|---|---|
| O1 | tight padding (R1) | 42% fewer pixels for a 16:9 frame | both |
| O2 | precision and backend (R2, R3) | float16 halves the GPU's memory traffic; int8 cuts the CPU's work | both |
| O3 | uint8 input in the graph: the model takes the canvas's RGBA bytes and casts, scales and transposes them itself | no per-pixel JavaScript loop, and a third of the bytes to upload (3.8 MB a frame against 11.3 MB of floats) | both (WebGPU may run those operators on the CPU: its log says) |
| O4 | WebGPU graph capture, input and output kept in GPU buffers (`Tensor.fromGpuBuffer`, `preferredOutputLocation`) | replays the recorded GPU commands instead of sending each operator from JavaScript | WebGPU |
| O5 | preprocessing on the GPU: a compute shader reads the video frame (`importExternalTexture`) and writes the model's input buffer | no copy to the CPU and back | WebGPU, **only if** preparing and uploading the input still take ≥ 20% of a detection after O3 and O4 |
| W1 | WebAssembly on 1 thread against ONNX Runtime's default | what threads give on a phone's mix of big and small cores | WebAssembly |

Left for [Later](#later): the native WebGPU build of ONNX Runtime Web
(`jspi`, 17.1 MB against the JSEP build's 28.2 MB), TopK inside the graph
to shrink what is read back, a 1280 × 720 stream with no resize.

---

## What we expect (written before any run)

Guesses, to be checked:

| Measure | Expect |
|---|---|
| R0: ONNX float32 square against PyTorch | the same boxes; AP equal to 3 decimals |
| R1: tight against square | AP within 0.01; a few confidences move by up to about 0.1; perhaps one date near the threshold crosses it |
| int8, MinMax | AP 0.5 to 2 points down, real-val-small the most |
| float16 on WebGPU | as float32, AP within 0.005 |
| the app against Ultralytics on the same photos | the same dates found; confidences within about 0.02 (different resize filters) |
| the laptop's Chrome, WebGPU, float16, 736 × 1,280 | 20 to 40 ms a detection |
| the phone, WebGPU, float16 | 60 to 200 ms |
| the phone, WebAssembly, float32 / int8 | 600 to 1,500 ms / 1.3 to 2 times faster |
| preparing the input (canvas and the JavaScript loop) on the phone | 10 to 30 ms, so O3 likely kept on WebGPU |
| detections a second while still | 3 to 5 on WebGPU (the rest rule caps it at 5), about 1 on WebAssembly |
| holding still to boxes | 150 ms of stillness plus one detection |
| 5 minutes held on a pack | 10 to 30% slower by the fifth minute (heat) |
| first load and warm-up over the Wi-Fi | under 10 s: a 28 MB runtime (uncompressed), a 3 to 11 MB model, shaders compiled |

---

## What we start from

| Fact | Where it comes from | What it means for M2c |
|---|---|---|
| The M2b detector `n1280-ft-s0` at `threshold_store` 0.1641; its `last.pt` (5.6 MB) on the laptop, its SHA-256 in `m2b.py` | M2b S12 to S14 | exported from the laptop's copy, its hash checked first |
| Every score so far pads square (`rect=False`) | M1 D3, `detector.predict` | tight padding is a departure: R1 |
| 175 ms a photo on the laptop's CPU (PyTorch, square); 54 ms a pass on `cn4`'s 24 CPUs | M2b S13 | what the browser is compared with |
| v1-val: 34 photos, long side 1,600 px (median), exported upright and numbered | M2b S6, S13 | the store's bench, laptop only |
| The laptop on 2026-11-22: Node 25.9.0, npm 11.12.1, Chrome and Safari, OpenSSL 4.0.3; no `adb`, no `mkcert` | `--version` | the app builds here; only the fallbacks would need a tool installed |
| The laptop's venv: Ultralytics 8.4.177, torch 2.11.0, onnx 1.23.2, **onnxruntime 1.31.0**, though the `read` extra pins 1.31.1; no `onnxslim` | `pip` | S1 moves ONNX Runtime to 1.31.1 (the web runtime's version too) and adds `onnxslim`, by the constraints method |
| Ultralytics' export names its dynamic axes `batch`, `height`, `width`; its `LetterBox` pads with 114, resizes with `INTER_LINEAR` and centres by `round(d ± 0.1)`; prediction's NMS is at IoU 0.7, 300 boxes at most; `YOLO_AUTOINSTALL` turns its installing off | Ultralytics 8.4.177's source | what the TypeScript mirrors and is tested against |
| Ultralytics' ONNX backend sets no thread count | `nn/backends/onnx.py` | on the cluster ONNX Runtime would start 48 threads and fail to pin them (CLAUDE.md, M3 S1): the job sets them as `scripts/crop_experiment.py` does for RapidOCR |
| `onnxruntime` ships `transformers.float16` (with `keep_io_types`) and `quantization` (MinMax, Entropy, Percentile, Distribution) | the venv | float16 and int8 need no new package |
| `onnxruntime-web` 1.31.1 (2026-10-26, MIT): the WebGPU build loads the JSEP WebAssembly (28.2 MB uncompressed), the plain build 14.2 MB; it has `freeDimensionOverrides`, `enableGraphCapture`, `Tensor.fromGpuBuffer` and `preferredOutputLocation` | the npm package | O3 to O5 are possible with this runtime |
| Scanify's frontend: React 19.2.8, TypeScript 6.0.3, Vite 8.2.2, Tailwind 4.3.3, `plugin-basic-ssl` 2.3.0, Vitest 5.0.3, oxlint 1.82.0; an HTTPS dev server on the Wi-Fi with `/api` proxied to its backend; its camera asks for 1920 × 1080 | its `package-lock.json` and `vite.config.ts` | the same stack and serving, so M2c's detector module can move there |
| The cluster didn't answer from home on 2026-11-22 (`ssh` to its campus address timed out) | `ssh` | S1's job waits for campus (or a VPN); everything else is laptop and phone |
| M3: S1 to S5 done, S6 next; GPU jobs `73946` and `73947` held | M3's plan | they keep their submit time while M2c runs |

---

## Decisions

Each with its reason. **(new)** marks a choice not written down before;
**(changes …)** a departure from an earlier one.

1. **(changes CLAUDE.md's milestone order) M2c comes before M3 resumes**,
   time-boxed to three sessions. M3 waits at S6; its held GPU jobs keep
   their place, since a held job keeps its submit time (CLAUDE.md).
   CLAUDE.md's milestone table gains M2c, with the owner's OK (S5).
   **Changed 2026-11-23, after S1, at the owner's word:** M2c and M3
   alternate. S2 to S5 need only the laptop and the phone, and none of
   them feeds M3 (M3 reads with its own Python ONNX path, not the app);
   M3's slices need the cluster, reachable only on campus. So campus
   days go to M3, home days to M2c. The three-session box still holds,
   counting M2c's sessions only.
2. **(new) A React web app, not a native app**, built like Scanify's
   frontend (the same libraries and versions, the same HTTPS and `/api`
   pattern): one codebase for Android and iPhone, and a detector module
   that can move into Scanify. Written here from nothing; no Scanify code
   is copied in (CLAUDE.md).
3. **(changes M5's order, not its goal) The phone's stack is built now,
   for the detector.** M5 planned it for M4's checker; the checker later
   takes the detector's place in the same loop. M5's "done when" stands.
4. **(new) The M2b detector, unchanged**: no retraining, no new threshold
   (0.1641). M2c measures faithfulness and speed; accuracy is M2b's.
5. **(new) Private.** Served from the laptop to the owner's phone behind a
   key; never a public address, never a tunnel. The model was trained on
   the retailer's photos, and the permission of 2026-10-28 doesn't cover
   publishing (CLAUDE.md). Screen recordings show packs: none is
   committed, and the README gets numbers only.
6. **(new) The model always sees 1,280 px on the long side**, from a
   1920 × 1080 stream. A frame whose long side is under 1,280 px gets a
   warning; 1280 × 720 is fine.
7. **(changes M1 D3, if R1 says) Tight padding** for the camera, measured
   by R1 first.
8. **(new) Throttle and gate, not debounce**: one detection at a time on
   the newest frame, a stillness gate, rest when nothing changes, boxes
   held by a tracker.
9. **(new) WebGPU first, WebAssembly as the fallback**, the precision for
   each by R2; ONNX Runtime Web 1.31.1, the version the venvs run.
10. **(new) One dynamic-shape file per precision, its shape pinned per
    session**; fixed-shape files only if pinning fails on WebGPU.
11. **(new) Calibration on training photos only; validation photos only
    score**, and the rules choose among fixed variants on them, as M2b's
    rules chose on v1-val. Store set v2 isn't touched.
12. **(new) Ultralytics is the reference** for letterbox, NMS and boxes:
    the TypeScript is tested against fixtures made by Ultralytics' own
    functions, and the app's boxes against Ultralytics' on the same
    photos.
13. **(new) Measured on the built app on the owner's phone**, unplugged,
    other apps closed, after 5 minutes' rest before each bench. The records
    say which phone and browser.
14. **(new) Who writes what** (the owner's choice for M3 on 2026-11-13,
    kept): Claude writes the letterbox, the NMS and the tracker against
    tests written first, then walks the owner through them line by line.
    The tracker's IoU match is offered as the owner's exercise, not
    pushed.
15. **(new, 2026-11-24, S2; changes Decision 13's "the owner's phone")
    Android with Chrome decides**, because the store staff mostly use
    Android (the owner). R2's speed half, R3 and R4 are read on **the
    owner's OnePlus 13** (S2's record): no staff phone can be borrowed now,
    and the owner chose to go on with it. It is a 2025 flagship, so its
    numbers are the best case; a staff-like phone's test is in
    [Later](#later), run if the app meets trouble on the staff's phones.
    The owner's iPhone is a second reading, never a rule's. On an iPhone
    every browser runs on WebKit, Safari's engine, so Chrome there is
    Safari's engine in Chrome's frame; Android's Chrome is another engine
    (Blink) with another WebGPU (Dawn on Vulkan).
16. **(new, 2026-11-26, S3; changes the rest under [The frame loop](#the-frame-loop))
    Rest follows the gate**: after 3 detections in a row with no frame
    over T, one a second until a frame over T. The plan asked for 3 with
    the same boxes; on a still pack the boxes flicker (the count changed
    between 76% of detections in S3's first soak), so that rested 5 s in
    300. Whether a detection's boxes matched the one before is still
    recorded. The owner's choice.
17. **(new, 2026-11-25, S3) The gate's copy is drawn on a canvas on the
    GPU**, after a gate bench on the phone, which the owner chose to run
    before pinning T: 11.5 ms a frame, against 64 ms on a canvas kept on
    the CPU (95% of the page's main thread). Still over S3's 2 ms, so the
    gate reads every other frame.
18. **(new, 2026-11-27, S4; changes how R4 reads the letterbox step) The
    letterbox on a GPU canvas is judged by S2's check against Ultralytics,
    not by R4's equal boxes**; R4's 10% speed half stands. The GPU canvas
    resizes differently from the CPU canvas: on the laptop its input was
    within 1 to 2 grey levels of Ultralytics' `LetterBox`, the CPU canvas's
    15 to 20, so it is closer to the reference, yet 2 of 20 boxes at 0.1641
    differed from the CPU canvas's on 5 photos. R4's box half was written
    for steps that change no number (O3 and O4 gave identical boxes); a
    step that changes the resize is what S2's check judges (every date the
    reference finds but at most one, AP within 0.01). The owner's choice.

---

## Code layout

```
src/bestbefore/
  live.py           the export (every variant, later O3's uint8 input), the calibration
                    reader, the manifest, the rules R0 to R4 and G, the report's tables
  detector.py       changed: predict(..., rect=False) passes rect through (default unchanged)
scripts/
  export_live.py    the M2b detector to ONNX, every variant, the manifest (laptop)
  live_parity.py    PyTorch and ONNX over the real sets, square and tight (the laptop's
                    v1-val smoke, the CPU job)
  live_fixtures.py  letterbox and NMS fixtures from Ultralytics' own functions, for Vitest
  live_api.py       the API on 127.0.0.1: models, bench photos, records
  score_live.py     the app's parity benches against the job's records
  m2c_report.py     results/m2c_live.md from the records
slurm/
  m2c_cpu.sbatch    task `parity`: int8 calibration, then scoring
web/                the React app (above); web/public/ort/ copied from node_modules
tests/
  test_live_export.py, test_live_parity.py, test_live_rules.py, test_live_api.py,
  test_live_fixtures.py, test_score_live.py
web/src/**/*.test.ts   Vitest
experiments/
  m2c-export/       does the exported model see what PyTorch sees? (R0 to R2)
  m2c-phone/        how fast is it on a phone, and what makes it faster? (R2 to R4, the loop)
```

Model files go to `$CHECKPOINT_DIR/m2c/` (`*.onnx` is already
gitignored); records to `results/runs/m2c/`. `.gitignore` gains
`web/node_modules/`, `web/dist/` and `web/public/ort/`. `.env.example`
doesn't change: the API reads the variables already there. Tests that need
`onnxslim` skip on the cluster, which only scores (as the `ensemble`
extra's do).

---

## Slices

Numbered in the order to do them: the one slice with a cluster job comes
first. Every slice records its findings here when done, and says what
wasn't checked.

### S1. The model files and their parity · laptop, then a CPU job · *submits 1 CPU job*

**Goal:** the ONNX files made and hashed; R0, R1 and int8's half of R2
read.

- The `live` extra (`onnxslim==0.1.99`) and `onnxruntime==1.31.1`,
  installed by the constraints method (CLAUDE.md), the dry run's report
  first.
- `export_live.py`: `last.pt`'s SHA-256 checked against `m2b.py`;
  `det-fp32.onnx`, `det-fp16.onnx`, `det-fp16-head32.onnx` and the
  manifest. The opset is the one Ultralytics picks, recorded.
- `detector.predict(rect=...)` and `live_parity.py`: Ultralytics predicts
  with the ONNX file as with the `.pt` (through its ONNX backend), so the
  letterbox and NMS are its own; one photo a call; ONNX Runtime's threads
  set to the job's CPUs.
- **The laptop's smoke**, v1-val only (34 photos a variant): PyTorch and
  ONNX float32 square (R0 on v1-val), ONNX float32 tight; int8
  calibration on 8 training photos.
- **The job** (`m2c_cpu.sbatch parity`; `short`, 24 CPUs, 32 GB, 2 h):
  the ONNX files copied from the laptop and their hashes checked first;
  int8 calibrated twice (MinMax, Percentile) on the 200 training photos;
  then the five real sets (570 photos) through 7 variants: PyTorch square;
  float32 square and tight; int8 MinMax and Percentile, square and tight.
  The int8 files come back to the laptop with the records.
- Before submitting: `squeue --me` (5 jobs submitted per user, other
  projects' included; M3's two held jobs count).
- **Tests** (`tests/test_live_export.py`): the manifest's hashes match the
  files; float16 and int8 keep float32 inputs and outputs and the dynamic
  axes; the int8 exclusion list is not empty and names only Detect-head
  nodes; `rect=True` reaches Ultralytics; the calibration reader gives
  tight-letterboxed inputs from training photos only.

**Check:** R0 passes on v1-val on the laptop before the job is submitted.
**Commits:** `docs: m2c plan` (this file, first),
`feat: m2c model export, float32 float16 and int8`,
`feat: m2c parity of the exported detector, and its cpu job`,
`docs: m2c plan, S1's export`

**Record (2026-11-22, laptop part).** Done at home; the cluster couldn't
be reached, so `slurm/m2c_cpu.sbatch parity` is written, checked
(`bash -n`, a 2-photo dry run of its steps on the laptop) and not
submitted. Its campus steps are in its header.

Departures:

- **float16 through ONNX Runtime's `OnnxModel` wrapper**
  (`onnxruntime.transformers.onnx_model`), not `float16`'s converter
  alone. Alone, with the head's decoding kept float32, the converter made
  an invalid model: two Cast nodes named
  `/model.23/Reshape_6_output_0_cast_to_fp32_node`. It casts every kept
  node's output back to float16 and the next kept node's input to float32
  again, so the decoding's tensors still passed through float16 between
  its nodes (box coordinates near 1,280 px in 1 px steps), and a tensor
  two kept nodes read got two Casts of one name. The wrapper runs the same
  converter (`keep_io_types`, types by symbolic shape inference), then
  removes Cast chains and Casts to the type they already hold. Both float16
  files are made this way. `live.float16_problems` checks the result: no
  repeated name, no decoding node reading or writing float16, and only
  Casts gone. Six went: the decoding's own Casts of `Range`'s integers to
  float32 were merged with the converter's into one Cast each, the same
  computation under the converter's name (listed in the manifest).
- **Batch is dynamic too**: Ultralytics names `batch`, `height` and
  `width` whenever `dynamic=True` (the table above said batch 1). The app
  pins all three per session (`freeDimensionOverrides`).
- **The calibrator reads one photo at a time** (ONNX Runtime's
  `CalibStridedMinMax` at 1, which strides any method despite its name).
  ONNX Runtime 1.31.1's Percentile calibrator has no
  `max_intermediate_outputs` (only MinMax has): it keeps every photo's
  every tensor until the last, and it stacks them with `np.asarray`, which
  fails on tight photos of different shapes. For MinMax the result is the
  same as all at once; for Percentile the histogram's bins are set by the
  first photo and widened by later ones, so the photos' order (the seed's)
  is part of the result.
- **The laptop's 8-photo int8 files live in `$CHECKPOINT_DIR/m2c/smoke/`**,
  with their own manifest, so they can't be taken for the job's
  200-photo files.
- **The threshold is M2b's record, unrounded**: 0.16406640 from the run's
  `suite-m2b.json`, as M2b scored store set v2 (0.1641 rounded).
  `live.threshold` reads it.
- **PyTorch predicts from M2b's own `last.pt`**, read only, its hash
  checked by `m2b.last_pt`. The copy in `$CHECKPOINT_DIR/m2c/` is the
  export's, since Ultralytics writes the `.onnx` beside the `.pt`. Same
  file, same hash.
- **Records** go to a run folder, `results/runs/m2c/<time>-<name>/`
  (`run.json`, a `parity-<set>-<variant>.json` per set and variant,
  `rules.json`). v1-val's records go there too, not to `data/store/runs/`
  where `sets.runs_root` puts store runs: they hold boxes as numbers,
  photos by set and index, as the plan says.
- **Tests in three files**, so each commit adds whole files:
  `test_live_export.py` (the export, float16, int8, the manifest, the
  calibration photos, ONNX Runtime's threads), `test_live_parity.py`
  (`rect` reaching Ultralytics, the padding check, the rules read off a
  run, the job's steps) and `test_live_rules.py` (R0, R1 and R2's accuracy
  half; the layout names this file, S1's list didn't). The rules are in
  `live.py` already, as the layout says.
- **One package more**: the dry run of `onnxslim==0.1.99` and
  `onnxruntime==1.31.1` (everything else held, onnxruntime taken out of
  the constraints) also brought `colorama==0.4.6`, onnxslim's dependency.
  The freeze changed by exactly those three lines; `pip check` is clean.

What was found:

- **The weights**: `last.pt`'s SHA-256 is `m2b.py`'s (`c839af8a…`), and so
  is its copy's. M2b's run folder is unchanged.
- **The export** (Ultralytics 8.4.177, torch 2.11, onnx 1.23.2, onnxslim
  0.1.99): opset **18**, Ultralytics' pick for torch 2.4 and later (its
  comment: ONNX Runtime's CUDA build has no Resize-19 or ReduceMax-20
  kernel, and its static int8 refuses opset 21 and up). IR version 8; 419 nodes;
  10.80 MB (Ultralytics prints MiB); `images [batch, 3, height, width]`
  to `output0 [batch, 8, anchors]`; metadata `names`, `stride` 32,
  `imgsz` [1280, 1280]. 2 s, 0.67 GB at peak. Ultralytics warns
  "no one-to-one head; using one-to-many outputs" (YOLO11 has none:
  normal) and, when passed `int8=False`, that `int8` is deprecated (now
  not passed). torch warns its TorchScript exporter (`dynamo=False`,
  Ultralytics' choice) is legacy.
- **The export isn't byte-reproducible.** Three exports of the same
  weights made 11.38, 10.92 and 10.80 MB files with identical nodes and
  weights, differing only in their shape annotations (`value_info`: 2,768,
  1,730 and 1,378 entries). The manifest pins the one kept
  (`det-fp32.onnx`, `5c755123…`).
- **The Detect head is module 23**, read from the node names. Its 172
  nodes: 60 in the box and class convolutions (`cv2`, `cv3`: 24 Conv, each
  SiLU a Sigmoid and a Mul) and **112 decoding**: anchors built from the
  input's shape (with height and width dynamic, they are computed in the
  graph: Shape, Gather, Range, Cast, Expand), the DFL (Softmax and its
  fixed Conv), the boxes scaled by stride (Sub, Add, Div, Mul), one
  Sigmoid for the classes and the Concat that writes `output0`. The list is
  in the manifest; `live.decode_problems` checks it against the graph
  (everything downstream of the convolutions is in it).
- **Ultralytics' own ONNX int8 takes another route**
  (`utils/export/onnx.py`): it quantises Conv, Gemm and MatMul only and
  excludes every other node. Ours follows the plan: everything but the
  decoding, so the backbone's and neck's Add, Mul, Concat and Resize are
  quantised too. R2 judges it.
- **float16**: `det-fp16.onnx` 5.61 MB, 421 nodes (the input's and
  output's Casts; `Range`, on the converter's default block list, is
  integer and needs none); `det-fp16-head32.onnx` 5.66 MB, 429 nodes, 16
  Casts. Both load in ONNX Runtime with float32 inputs and outputs and the
  float32 file's metadata. Their accuracy is S2's, in the browser.
- **int8, the laptop's smoke**: 8 photos (3 store, 5 ExpDate, list
  `c1c4fc21…`), the start of the job's 200 (a smaller draw is the
  start of the full one), letterboxed tight to 1,280 × 736 and 1,280 × 960;
  376 tensors calibrated. MinMax 4.2 s, Percentile 24.8 s (3.1 s a photo,
  its histograms), 4 threads; 1,210 nodes and 3.4 MB each; 1.47 GB peak
  resident, 2.09 GB footprint. **Swap grew from 1.58 to 2.22 GB during
  it** (macOS grew the swap file from 2 to 3 GB), so heavy work paused.
  Nothing after it grew swap; the full calibration is the job's. ONNX
  Runtime warns "consider pre-processing before quantization". Not run:
  its optimiser renames nodes, and nodes are excluded by name; the export
  already fused BatchNorm and onnxslim folded constants.
- **Percentile's range is symmetric**, [−t, t]: its histogram is of
  absolute values, ONNX Runtime's default. With uint8 activations that
  gives half the levels to negatives that SiLU hardly reaches (−0.28 at
  least). It may cost Percentile against MinMax; R2 reads it.
- **An untrained YOLO11n breaks the Percentile calibrator**: numpy can't
  cut 2,048 float32 bins over a range of 1e-6. Ultralytics initialises the
  head so an untrained net's class sigmoids sit at 0.5 and its DFL logits
  at 2.0. The tests add seeded noise to the tiny model's weights. The bins
  are made from the first photo only, and the job's draw starts with the
  smoke's first photo, which passed.
- **Ultralytics' ONNX session** runs `intra_op_num_threads` 0 (every
  core) unless set, and torch drops to 7 threads at the first prediction
  (CLAUDE.md). `live.ort_threads` sets every ONNX Runtime session made
  inside it, Ultralytics' and the calibrator's; the parity run sets
  torch's after the warm-up, and checks both.
- **R0 passes on v1-val** (run `20261122-192848-m2c-parity-laptop-v1val`,
  34 photos, 4 threads): PyTorch square and ONNX float32 square find the
  same 60 of 63 dates and 29 of 29 expiries, with AP@0.5 for dates 0.9596
  both (the difference 0.000000) and confidences at most 2.6e-6 apart.
  0.9596 is M2b S13's laptop AP. At 0.1641 the laptop keeps 77 boxes (17
  false, precision 0.779); M2b's suite on the cluster (16 photos a call)
  kept 76 (precision 0.790): one box near the threshold differs between
  the two runs. That is why R0 compares on one machine.
- **R1, an early look on v1-val only (not its reading)**: tight padding
  gives AP 0.9409 (0.0187 under square's, over R1's 0.01), 59 dates (one
  fewer, allowed) and **28 of 29 expiries**: one expiry falls from 0.458
  to 0.075, far from the threshold. So on v1-val tight fails R1 twice.
  R1 needs every set, so the job's reading will very likely keep square
  padding; it is read on the job's records, as decided. Tight inputs were
  1,280 × 736 (19 photos), 1,280 × 960 (12) and 960 × 1,280 (3). Of the 62
  dates both paddings box, confidence moved by a median of −0.003, 9 by
  more than 0.1 (at most −0.54 and +0.20).
- **Speed on the laptop's CPU** (Apple M2, 4 threads, one photo a call,
  the photo already decoded; median over v1-val): PyTorch 166 ms, ONNX
  float32 square 134 ms, tight 76 ms (43% less). In the 2-photo dry run,
  int8 took about 70 ms square, half of float32 (too few photos to be a
  measure).
- **The dry run of the job's steps** (`check`, then 7 variants × 5 sets
  × 2 photos with the smoke's int8 files, then `rules`): 70 passes in 14
  s, 0.80 GB at peak; its folder deleted after, since 2 photos read no
  rule. Ultralytics read the class names from the int8
  files' copied metadata. macOS's `find` refuses `-perm /077` (GNU's
  syntax; the cluster's `find` is GNU, as in `m3_cpu.sbatch`). A v1-val
  record is about 60 KB, so the job's 35 should come to about 10 MB.
- **Tests**: 18 in `test_live_export.py` (a tiny YOLO11n exported at
  64 px through the same functions; skipped where onnxslim is missing), 4
  in `test_live_parity.py`, 3 in `test_live_rules.py`. Each commit alone
  in a worktree with absolute `.env` paths: the export commit 18 passed;
  the parity commit 77 passed (with the detector, `eval_detector`,
  `score_suite` and ensemble tests, everything that fakes or calls
  `predict`). None skipped. The full suite wasn't run, to spare the
  laptop.

Not checked, the job's part: R0 on real-val, real-val-rot, real-val-small
and real-val-rot180; R1, which needs all five sets; int8's accuracy (R2's
accuracy half) and the 200-photo calibration's time and memory on a
compute node (the laptop's 3.1 s a photo for Percentile suggests about 10
minutes); ONNX Runtime's threads inside a job's cgroup; whether the
cluster's main venv imports everything the job needs (onnxruntime 1.31.1
and onnx 1.23.2 came with M3 S1). Not S1's: float16's accuracy (S2).
(All but float16's were checked the next day, below.)

**Record (2026-11-23, the cluster).** From campus. The state was checked
first, not taken from the laptop's record. The owner had already made and
pushed S1's three commits (`6f55f32`, `98e5683`, `33a306a`). The
cluster's checkout was clean at `5240800` (M2b S14) and was pulled
fast-forward to `33a306a`. The pull touched nothing M3's two held jobs
run: they are placeholders that start `slurm/m3_chain.sh`, not yet
written. The three float files and the manifest went by `scp` into an
owner-only `$CHECKPOINT_DIR/m2c/` (folder 700, files 600), their SHA-256
the laptop's. On the login node first: `export_live.py check` passed (the
cluster's own M2b `last.pt` matches `m2b.py`), and the M2c and detector
tests gave 31 passed and 11 skipped (the onnxslim ones, as designed), in
95 s. Before submitting, our only jobs were M3's two held ones, and
`short` had 2 idle nodes. **Job 74149** (`m2c_cpu.sbatch parity`,
commit `33a306a`) started at once on `cn1` and completed in 22 min 15 s,
exit 0, with nothing in its run or model folder open to others. The
records and the int8 files came back with `tar` over `ssh`. On the laptop
all five files pass `check`; the cluster's manifest keeps the float
entries as they were and adds `int8`; and the laptop's `rules` on the
fetched records gives the job's `rules.json` byte for byte.

AP@0.5 for dates, and dates found at 0.1641 (expiries on v1-val), run
`20261123-165009-m2c-parity`:

| Set | PyTorch square | float32 square | float32 tight | int8 square | int8 tight | int8-pct square | int8-pct tight |
|---|---|---|---|---|---|---|---|
| v1-val (63 dates, 29 expiries) | 0.9601 · 60 · 29 | 0.9601 · 60 · 29 | 0.9404 · 58 · 27 | 0.9306 · 57 · 27 | 0.9117 · 52 · 25 | 0.9402 · 58 · 28 | 0.9170 · 56 · 28 |
| real-val (187) | 0.9949 · 187 | 0.9949 · 187 | 0.9949 · 187 | 0.9884 · 187 | 0.9921 · 187 | 0.9944 · 187 | 0.9949 · 187 |
| real-val-rot (187) | 0.9850 · 187 | 0.9850 · 187 | 0.9872 · 186 | 0.9730 · 186 | 0.9757 · 186 | 0.9813 · 187 | 0.9829 · 187 |
| real-val-small (186) | 0.9845 · 184 | 0.9845 · 184 | 0.9845 · 184 | 0.9819 · 183 | 0.9819 · 183 | 0.9808 · 184 | 0.9808 · 184 |
| real-val-rot180 (187) | 0.9504 · 177 | 0.9504 · 177 | 0.9492 · 175 | 0.9444 · 177 | 0.9403 · 178 | 0.9410 · 177 | 0.9315 · 177 |

The readings:

- **R0 passes on all five sets.** PyTorch square and ONNX float32 square
  find the same dates with the same AP (difference 0.000000 on each set),
  the same false boxes, and confidences at most 4.9e-6 apart. PyTorch's
  counts are M2b's suite's on the cluster (v1-val 60 of 63, 29 of 29
  expiries, 0.4706 false boxes a photo; real-val 187 of 187, 0.097). Its
  v1-val AP is 0.9601 here and 0.9596 on the laptop.
- **R1 fails, so the app pads square.** On v1-val tight is 0.0197 under
  square, with 2 dates fewer and **2 expiries lost**; on real-val-rot180 it
  finds 2 dates fewer. It passes real-val (AP within 0.0001), real-val-rot
  (0.0022 up, one fewer) and real-val-small (the same). Decision 7's
  departure from M1 D3 isn't taken, and O1 is out: a 16:9 frame keeps
  square padding's 78% more pixels than tight. The laptop's early look at
  v1-val found 59 dates and 28 expiries; the cluster's is 58 and 27.
- **R2's accuracy half keeps float32 on WebAssembly.** At square (R1's
  padding), int8 (MinMax) fails on v1-val (0.0295 under, 3 dates fewer)
  and on real-val-rot (0.0120 under). Its fallback, int8-pct, also fails
  on v1-val (0.0199 under, 2 dates fewer). Both pass the other sets. So
  R2 gives float32, whatever their speed: S4's WebAssembly benches need
  float32 only (and W1's threads), and int8's speed on the phone is at
  most a side reading. Tight, int8 is worse still on v1-val (52 dates, 25
  expiries).
- **Against what we expected**: R0 as written. R1 not: AP within 0.01
  was expected, and v1-val lost 0.020 and 2 expiries. int8 MinMax was
  expected to lose 0.5 to 2 points, real-val-small the most. It lost 2.9
  on v1-val, the store's photos, the set that matters most, and 0.3 on
  real-val-small. Percentile, its range symmetric, did better than MinMax
  on three sets of five (v1-val, real-val, real-val-rot), at both
  paddings.

The job's other findings:

- **The int8 calibration** on `cn1`, 24 threads: 200 photos (80 store,
  120 ExpDate, list `06fcc883…`) letterboxed tight into 20 shapes, from
  704 × 1,280 to 1,280 × 1,280. MinMax took 62 s and Percentile 266 s (1.3
  s a photo against the laptop's 3.1), 2.65 GB at peak. Both calibration
  sessions and every scoring session were made with 24 threads, checked
  by the scripts, and no pinning error was printed. `det-int8.onnx` is
  `b72ff22c…`, `det-int8-pct.onnx` `5b1179aa…`, each 3.4 MB and 1,210
  nodes.
- **The job's log lost its first lines**: `tee /dev/stderr` reopened it
  truncated. A job's stderr is its log, and on Linux `/dev/stderr` opens
  the file afresh; reproduced on `mn1`. The laptop's dry run couldn't show
  it, its stderr a terminal. The start line, the check and the
  calibration's output went; the manifest holds the calibration's record.
  Fixed: the run's output now tees into a `mktemp` file (CLAUDE.md, HPC).
- **The ONNX passes took about 4 times too long**: 164 to 327 ms a call
  over the six ONNX variants (float32 square 281 to 301), against
  PyTorch's 33 to 42 ms. Three short diagnosis jobs (74150 to
  74152, [experiments/m2c-cpu-threads](../experiments/m2c-cpu-threads/README.md))
  found ONNX Runtime's inference takes 52 ms. Ultralytics' torch pre- and
  post-processing around it crawled, because libgomp's spinning threads
  fight ONNX Runtime's on the same CPUs. With `OMP_WAIT_POLICY=PASSIVE`,
  or torch on one thread, a call takes 72 ms. Denormal floats were tested
  and ruled out. The job script now exports `PASSIVE`. Job 74149's ONNX
  times aren't quoted anywhere; its boxes and scores stand. Ultralytics
  set torch to 8 threads at its first prediction in a 24-CPU job, its rule
  now measured (CLAUDE.md).
- The run folder holds 35 parity records, `run.json` and `rules.json`:
  4.6 MB (estimated about 10).

Not checked: float16's accuracy (S2, in the browser). Nor ONNX Runtime's
times over whole sets with `PASSIVE` (the diagnosis timed 12 real-val
photos, float32 square), nor int8's speed on x86, which R2 no longer
needs. The full test suite wasn't run; the cluster ran the M2c and
detector tests, and the laptop each commit's.

### S2. The app: camera, worker, boxes, benches · laptop, then the phone

**Goal:** boxes on the phone's camera, and the app's detector matching
Ultralytics'.

- `web/` from Vite's React and TypeScript template, the pinned libraries,
  oxlint and Vitest; a `prepare` script copies ONNX Runtime Web's
  WebAssembly files into `public/ort/`.
- `live_api.py` and Vite's config: HTTPS, the `/api` proxy, the key,
  COOP and COEP, the MIME types, `--with-store`.
- `live_fixtures.py` writes `web/src/detector/__fixtures__/`: Ultralytics'
  letterbox sizes and pads for 1920 × 1080, 1080 × 1920, 1600 × 1200,
  1280 × 720 and 640 × 480 frames, tight and square; its NMS and
  `scale_boxes` on made-up outputs with planted boxes (overlapping, two
  classes, ties, one under the threshold). Numbers only, no photo.
- `src/detector/` and a first Live page (one detection at a time, boxes
  drawn, no gate yet), and the Bench page.
- **First in the laptop's Chrome**: the model loads and runs on WebGPU and
  on WebAssembly, and the log names any operator that falls back to the
  CPU; parity benches of v1-val (`--with-store`) and real-val at float32
  and float16 on the laptop's GPU.
- **Then the phone**: open it, accept the certificate, allow the camera;
  boxes on packs at home; parity and speed benches of real-val for each
  backend and precision. Which phone and browser, WebGPU or not,
  `shader-f16` or not: recorded.
- **Tests**: Vitest, the letterbox and NMS against the fixtures, boxes
  mapped back to the frame and on to the screen; pytest
  (`tests/test_live_api.py`), the key required, no path out of its folder,
  v1-val refused without `--with-store`, a record written with the commit
  and the manifest, the headers.

**Check:** in the laptop's Chrome (WebGPU, float32), the app finds on
real-val every date Ultralytics' ONNX float32 finds at 0.1641 but at most
one, with AP within 0.01. Otherwise the resize is the first suspect:
a resize matching `INTER_LINEAR` in the worker, recorded.
**Commits:** `feat: m2c api for the phone app`,
`feat: m2c web app, the detector in a worker`, `feat: m2c bench mode`,
`docs: m2c plan, S2's app on the phone`

**Record (2026-11-24, laptop and two phones).** A home day. The state was
checked first: `main` clean at `43eee59`, no `web/` yet, and
`export_live.py check` passed for the three float files.

Departures:

- **The parity bench pads square, its session pinned at 1 × 3 × 1,280 ×
  1,280.** The Bench text above (each photo letterboxed to its own shape,
  a session without a pinned shape) was written for tight padding; R1
  kept square, so every photo and every camera frame reaches the model at
  one shape, and the parity and speed benches share it. The reference is
  the job's `parity-<set>-fp32-square.json`.
- **`score_live.py` and `tests/test_score_live.py` came in S2, not S5**:
  S2's check can't be read without them. It rescores the job's reference
  from its own per-photo boxes on the same photos, by the same code; on
  whole sets that gives the job's AP to 1e-6 (its boxes are kept to 5
  decimals), and a record of the first N photos is compared on those N.
- **Vite 8.2.6, not Scanify's 8.2.2** (the owner's choice, 2026-11-24):
  `npm audit` flagged 8.2.0 to 8.2.3 (GHSA-rq7h-c2jc-7f22, files outside
  `server.fs.allow` served when project paths match system paths;
  GHSA-vfpm-58rq-9qcg, the first 4 bytes of files outside `server.fs`
  through `?vite-wasm-instance`; GHSA-9jrq-w75r-8gcw; and three in
  `launch-editor`), and the plan puts the dev server on the Wi-Fi. 8.2.6
  (2026-11-09) is the 8.2 line's patch outside that range; `npm audit`
  then finds 0. Scanify's frontend is still on 8.2.2 with the same dev
  server on the Wi-Fi: Scanify's to fix. Four more exact pins, from
  Scanify's lockfile, which the plan didn't list: `@tailwindcss/vite`
  4.3.3, `@types/react` 19.2.18, `@types/react-dom` 19.2.7, `@types/node`
  24.13.3.
- **Two pixel checks in the parity bench**, so a box that moves can be
  blamed on the resize or the runtime: every photo's decoded pixels
  against PIL's (the API sends the SHA-256 of `load_upright`'s RGB), and
  the first 5 photos' model input against Ultralytics' `LetterBox`, value
  by value (`/api/bench/<set>/<i>/input`, raw RGB).
- **Store photos go to the laptop's browser only, even with
  `--with-store`.** Vite's proxy adds the browser's address
  (`X-Forwarded-For`, its `xfwd`), and the API serves a store set only
  when the last hop is a loopback address. Checked: from the laptop's own
  Wi-Fi address v1-val is refused (403), even with a forged
  `X-Forwarded-For: 127.0.0.1` (the proxy appends the real one), and the
  config lists real-val only. The phone's session ran the API without
  `--with-store`, so v1-val wasn't loaded at all.
- **What the phone work needed, beyond the plan**: an `error` record kind
  (a failure on the phone is otherwise seen by no one); `HEAD`; a bench
  queue in the URL (`queue=kind:set:variant:backend[:threads],...`, with
  `limit=`, `checks=` and `rest=` seconds before each speed bench,
  Decision 13), `run=1` taken out of the address once it starts; a screen
  wake lock while benches run; bench photos fetched past the browser's
  cache, so the API's log is the bench's progress when a tab dies; and a
  **Save numbers** button on Live that posts a `live` record (the camera's
  name and settings, the last 50 detections' stage times, the rate), until
  S3's live records. The app's build commit is baked in (Vite `define`)
  and every record carries it.
- **Commits regrouped**: `App.tsx` routes Live and Bench, so the bench page
  can't come in a commit after the web app's without a half-written
  `App.tsx`. The web app's commit holds both pages, and `score_live.py`
  gets its own (`feat: m2c parity benches scored against the job's
  records`). Plus `tests/test_live_fixtures.py`: the committed fixtures
  must equal what Ultralytics makes now, so an upgrade that changes its
  letterbox or NMS fails a test.

The app, as built:

- `web/` from Vite's React and TypeScript template (`create-vite` 8.2.0,
  which still writes Vite 7 and ESLint; its layout kept, oxlint in
  ESLint's place as Scanify does). Three tsconfigs: the worker gets the
  WebWorker type library, the page the DOM's. The template's
  `erasableSyntaxOnly` refuses TypeScript's constructor shorthand.
- **ONNX Runtime Web 1.31.1's `onnxruntime-web/webgpu` import isn't the
  JSEP build any more**: its bundle loads `ort-wasm-simd-threaded.
  asyncify.wasm` (28.7 MB, the native WebGPU provider). The JSEP build the
  plan measured (`jsep.wasm`, 29.5 MB, 28.2 MiB) is what the default
  `onnxruntime-web` import loads; the app uses it, for WebGPU and
  WebAssembly alike. `npm run prepare` copies its `.mjs` and `.wasm` into
  `public/ort/`; with `env.wasm.wasmPaths` set, the bundle imports that
  `.mjs` instead of its embedded loader, and the runtime's thread workers
  start from it, not from the app's worker. Vite still emits a second copy
  of the `.wasm` into `dist/assets/` (the bundle names it with
  `new URL(…, import.meta.url)`); it is never fetched, only 29.5 MB of
  disk.
- **Bundles**: the worker's chunk 416 kB (ONNX Runtime's JavaScript), the
  page's 207 kB. A first build put ONNX Runtime in the page's chunk too
  (615 kB), because the pages imported a name helper from `session.ts`.
- **Served**: `vite preview` over HTTPS on port 5180, every response with
  COOP `same-origin` and COEP `require-corp` (the worker's script
  included), `.wasm` as `application/wasm`, `.mjs` as `text/javascript`
  (checked with `curl`). Every page was cross-origin isolated, in the
  worker too. The dev server wasn't used for any measure.
- **Tests**: Vitest 85 (letterbox 51 on 15 frame shapes tight and square,
  NMS 12, the screen mapping 5, the rest), pytest 17 (`test_live_api.py`
  11, `test_score_live.py` 4, `test_live_fixtures.py` 2). Four deliberate
  breakages each failed a test: conf compared in float64, no 30,000 cut,
  IoU at 0.7 removing a box, `Math.round` for the pads.

What Ultralytics' functions taught the TypeScript (Decision 14):

- **Python's `round` takes halves to the even integer** (`round(500.5)` is
  500; `Math.round` gives 501): a 1,001 × 2,560 frame scales to 500.5 px.
  And `Math.round(-0.1)` is JavaScript's `-0`, where Python gives 0.
- **Torch compares a score with `conf` in float32**: a score of exactly
  0.3 isn't over a conf of 0.3, which a float64 comparison gets wrong.
  **torchvision's IoU** is float32 step by step, compared with the
  threshold as a double; predict imports torchvision, so Ultralytics uses
  its NMS. On equal scores the earlier anchor is kept.
- **`non_max_suppression` takes any output whose last axis is 6 for an
  end-to-end model** (`[1, 300, 6]`) and only filters it: a made-up output
  of 6 anchors gave nothing. The model's 33,600 never meet it.
- **The 30,000-candidate cut** decides the result only when the boxes
  past it would survive NMS; the fixture piles 30,000 high scores in one
  place and spreads 3,600 low ones, so 4 boxes are kept with the cut and
  300 without.
- **A perfect detector's AP is 0.995**, not 1: the AP integrates its 101
  recall points by trapezoids down to a last point of precision 0
  (Ultralytics' way, `detection_eval.average_precision`).
- Python's `http.server` turns a leading `//` into `/` itself.

What was found, the laptop's Chrome (Chrome 158, arm, 8 cores, 8 GB;
WebGPU on Apple `metal-3` with `shader-f16`; ONNX Runtime 1.31.1, 4
WebAssembly threads):

- **S2's check passes.** WebGPU float32 on real-val: AP 0.9949, 187 of 187
  dates, as Ultralytics' ONNX float32 finds them (lost 0, gained 0). The
  plan's fallback, a resize matching `INTER_LINEAR`, isn't needed.
- **v1-val**, WebGPU (34 photos, store photos in the laptop's browser
  only): float32 AP 0.9589 against Ultralytics' 0.9601, 59 dates against
  60 and **28 expiries of 29**. The lost one sat on the threshold:
  Ultralytics 0.1643, the app 0.1589, against 0.1641. float16 0.9589, 60
  dates, 29 expiries; fp16-head32 0.9602, 60, 29.
- **R2's accuracy half passes for float16 in this browser** on v1-val
  (60 dates against float32's 59, AP 0.0001 apart) and on real-val (AP
  equal, 187 each); fp16-head32 too. Its speed half fails here (below).
- **Decoding is PIL's**: every photo's decoded pixels had PIL's SHA-256
  (34 of 34, 165 of 165), with `colorSpaceConversion: "none"`.
- **The resize is close to OpenCV's, not equal**: on v1-val's first 5
  (1,600 → 1,280) the inputs differ by at most 1 grey level (23 to 40% of
  values differ); on real-val's first 5 (947 to 1,399 px, scaled 0.92 to
  1.35) by at most 15 to 20, a mean of 0.30 to 0.54. Box by box at IoU
  0.9, the confidences moved by a median of 0.0005 on real-val (95th
  percentile 0.045, at most 0.14; 55 of 468 pairs over 0.02) and 0.0033
  on v1-val (at most 0.073): more than the guess of "within about 0.02".
- **Every configuration gave identical boxes when repeated** (4 to 6 runs
  each, a 3-photo smoke).
- **Placement** (ONNX Runtime's verbose log while the session builds):
  float32 puts all 260 nodes (after its fusions) on WebGPU, WebAssembly
  all 236 on the CPU. **float16 sends 10 `Split` nodes to the CPU**, with
  21 `MemcpyFromHost` and 10 `MemcpyToHost` around them (fp16-head32: 288
  and 10). The same on the Android phone. JSEP doesn't take these
  float16 Splits; ORT also fused each SiLU into `QuickGelu` (77).
- **Speed** (parity benches, median ms a detection, model run in
  brackets; letterbox and float conversion about 5 ms each): WebGPU
  float32 89 (77), float16 120 to 136 (104 to 125), fp16-head32 114 to 127.
  **float16 is 35 to 60% slower than float32 here**, from the Split
  copies. WebAssembly float32 474 (449), on 3 photos. Session built in
  0.2 to 0.6 s, the first warm-up 0.1 to 1.7 s (shaders compiled once per
  browser, then cached).

What was found, the phones (the same built app, served over the Wi-Fi):

- **Android: a OnePlus 13** (the owner; model code CPH2691 by Chrome's
  client hints), a 2025 flagship, so the best case (Decision 15), Chrome 157,
  Android 17 by `platformVersion` (the user agent says "Android 10; K",
  frozen), 8 cores, 8 GB by `deviceMemory` (which stops at 8); WebGPU on
  Qualcomm `adreno-7xx` with `shader-f16`; 4 WebAssembly threads. The
  camera: "camera 3, facing back", 1920 × 1080 at 30 fps as asked,
  frames 1,080 × 1,920 upright, `focusMode` continuous in its settings but
  only `manual` in its capabilities, so the app's request wasn't made; 4
  cameras listed.
- **On Android the app matches Ultralytics on every backend and
  precision**: real-val 187 of 187, AP 0.9949, decoding PIL's on 165 of
  165. **float32 is reproducible across devices**: the model's input is
  the same value for value as the laptop's (Chrome resizes alike on both),
  and its boxes equal the laptop's to 5 decimals on 132 of 165 photos, the
  rest 0.00001 apart; WebGPU against WebAssembly on the phone the same.
  float16's boxes differ between the Apple and Adreno GPUs.
- **Android, live** (Save numbers, 35 to 50 detections each): WebGPU
  float32 216 to 224 ms a detection (model 147 to 148), **4.3 to 4.6
  detections a second**; WebAssembly 687 ms (617), 1.2 a second. The video
  frame to an `ImageBitmap` took 1 to 4 ms. Boxes on packs at home, a
  little behind the picture (the owner).
- **Android, parity benches** (median ms a detection, model in brackets;
  not rested, Decision 13 is for speed): WebGPU float32 221 (163), float16
  271 (219), fp16-head32 281 (228), WebAssembly float32 658 (625). The
  letterbox took 19 to 37 ms and the float conversion 12: **preparing
  the input is 31 to 49 ms**, over the guess of 10 to 30 (O3's case).
  Session built in 1.7 to 4.3 s (the 29.5 MB runtime fetched and compiled
  in it, per worker), the first warm-up 0.4 to 0.9 s, the model fetched in
  0.3 to 0.7 s.
- **Android, the rested speed benches** (Decision 13: unplugged, other
  apps closed, 5 minutes' rest before each, all 165 photos at the
  threshold after 5 warm-ups; median ms a detection, model in brackets):
  WebGPU float32 **228 (165)**, steady over the bench (the model's median
  165, 166 and 164 ms in its thirds; 152 to 177 from the 10th to the 90th
  percentile); WebGPU float16 **284 (235)**; WebAssembly float32, 4
  threads, **868 (817)**. As first readings, ahead of S4's: R3 would pick
  WebGPU (3.8 times faster), and R2's speed half fails float16 (25%
  slower, not 10% faster) while JSEP runs its Splits on the CPU. The
  rested WebAssembly bench ran 30% slower than the unrested parity bench
  an hour before (817 against 625 ms a run) and built its session in 9.1
  s against 1.7; why wasn't looked into (the phone's power state is the
  first suspect).
- **iPhone 11 Pro Max** (the owner's; iOS 26.6.1, Chrome for iOS 158,
  which runs on WebKit; asked "which phone", the owner first said Android
  Chrome, and the records said iPhone): 4 cores reported, WebGPU on
  Apple's GPU with `shader-f16`, cross-origin isolated, 2 WebAssembly
  threads. Camera "Back Triple Camera", 1920 × 1080 at 30 fps, no focus
  control offered (WebKit focuses by itself), 7 cameras.
- **On the iPhone WebGPU crashes the tab** ("Can't open this page", Chrome
  for iOS's page for a dead WebKit process): live WebGPU float32 ran 668
  to 710 ms a detection (model 618 to 644), 1.2 to 1.45 a second, then
  died after a few dozen; a parity bench died after 6 photos, twice; a
  WebGPU session built straight after a WebAssembly bench died within 3
  s, twice; started fresh, it died after 8 to 9 photos. **WebAssembly is
  stable**: a 30-photo speed bench twice, 905 and 946 ms a model run
  (median; 1.1 to 3.1 s for the first few), 920 to 965 ms a detection.
  The cause is unproven (iOS gives a page no memory numbers). The likely
  one: JSEP's GPU buffer cache keeps released buffers in size buckets
  whose caps add up to over 2 GB (25 MiB and 56 MiB, this model's feature
  maps at 1,280², among them), and each run makes and destroys a
  read-back buffer; a 4 GB iPhone's page limit is far lower. Its cache
  modes (`storageBufferCacheMode`) apply to the native WebGPU provider
  only, not JSEP. S4 takes it up.
- **Against what we expected**: the app's dates as Ultralytics' (one
  threshold-edge expiry on v1-val, none on real-val), confidences further
  apart than guessed. Phone WebGPU float16 was to take 60 to 200 ms: on
  Android float32 is 147 to 163 ms a run and float16 slower, 219. Phone
  WebAssembly float32, 600 to 1,500 ms: Android 625, iPhone 905 to 946.
  Live, 3 to 5 detections a second on WebGPU: Android 4.3 to 4.6 (no rest
  rule yet), iPhone none that lasts. The laptop's WebGPU (20 to 40 ms
  expected for float16 at 736 × 1,280) runs float32 at 77 ms at 1,280
  square; R1 took tight away.

How to run it (until S5's README): in one terminal, from the repo,
`source .venv/bin/activate && python scripts/live_api.py` (add
`--with-store` only for v1-val in the laptop's own browser); in another,
`cd web && npm run build && npx vite preview` (`npm ci` once first on a
fresh checkout: the lockfile's versions, and `prepare` copies ONNX
Runtime's files into `public/ort/`). The API prints a `laptop:` and a
`phone:` link, a new key each start; open them in Chrome, accept the
self-signed certificate once, and allow the camera. Ports 5180 and 8765
are fixed: one copy at a time. Benches start from the link too, e.g.
`#k=<key>&page=bench&queue=speed:real-val:fp32:webgpu&rest=300&run=1`.

Facts that belong elsewhere too (S5, with the owner's OK): the phone and
browser facts above, ONNX Runtime Web's import change, JSEP's float16
Split and its buffer cache, and the iPhone's WebKit, in CLAUDE.md's
lessons; the laptop's facts (Chrome 158 with WebGPU and `shader-f16`; the
desktop app's browser pane refuses a self-signed certificate, so laptop
benches run in Chrome) in its laptop notes; Vite's advisories in
Scanify's own notes.

Not checked: the iPhone's crash cause, its
parity on any backend, and Safari itself (the same WebKit as Chrome for
iOS, so expected alike); v1-val on WebAssembly (the laptop ran it on 3
photos of real-val only); tight padding in a browser (fixtures only: R1
chose square); the dev server's headers (only `vite preview` was
checked); heat and battery; **a staff-like Android phone** (Decision 15:
every Android number here is a flagship's, the best case; in Later). The full test suite wasn't run; each commit's tests were, in
a worktree.

### S3. The frame loop · laptop, then the phone

**Goal:** the loop under [How the app reads the camera](#how-the-app-reads-the-camera),
measured.

- `useStillness` (the gate, its own cost a frame recorded), the Motion
  page and rule G; `useDetections` (one at a time, 200 ms between starts,
  rest after 3 alike); `tracker.ts`; the overlay, the line of numbers, the
  wake lock, the stop while hidden; live records.
- **On the phone**: a Motion record (20 s still, 20 s moving) gives `T`;
  then a scan (one minute, five packs held about 5 s each) and a soak
  (5 minutes on one pack, the phone propped up).
- **Tests** (Vitest): the gate on made-up frames (a still frame with noise
  is still; a shift of a few pixels is motion; a change of brightness
  alone isn't); the scheduler (never two at once, frames dropped not
  queued, the 200 ms gap, rest after 3 alike, awake again on motion); the
  tracker (matching, smoothing, kept through one miss, dropped at two,
  cleared after 400 ms of motion).

**Check:** the gate costs under 2 ms a frame (median) on the phone, or it
runs on every other frame (recorded).
**Commits:** `feat: m2c stillness gate and detection scheduling`,
`feat: m2c box tracker and overlay`, `docs: m2c plan, S3's loop`

**Record (2026-11-25 and 26, the laptop and the Android phone).** Two home
days. The state was checked first: `main` clean at `4f2e435`, up to date
with `origin/main`; `npm ci` installed, with a warning that Vitest 5.0.3
asks for Node ^22.12, ^24 or ≥ 26 (the laptop has 25.9.0), and the 85
tests passed; `export_live.py check` found M2b's weights and the three
float files as the manifest says; ports 5180 and 8765 were free. The API
ran without `--with-store` throughout, so no store photo was served.

Departures:

- **Rest follows the gate, not the boxes** (Decision 16, the owner's
  choice, 2026-11-26). The first soak rested 5 s of 300 (below); now 3
  detections in a row with no frame over T start the rest, and `alike`
  is only recorded.
- **The gate's copy is drawn on a canvas on the GPU** (Decision 17).
  S3's first gate drew on a canvas kept on the CPU (`willReadFrequently`):
  on the phone 63 ms a frame, 95% of the page's main thread, 15 reads a
  second, every other camera frame unread. A **gate bench** was added to
  the Motion page to find a cheaper copy before T was pinned: four ways,
  3 s each, two rounds, the phone propped on a pack; a `gate-bench`
  record, a kind the API gained.
- **The gate reads every other frame** (`GATE_EVERY = 2`), by S3's own
  check.
- **The Motion page's cues say each action in the 3 s before its phase**,
  which aren't kept: the laptop's first record counted 0.4 s of "moving"
  before the hand moved, a third of the moving frames under the still
  p95, and rule G overlapped there (1.68 against 1.60). Each frame keeps
  its value against the frame before (`value1`) and the one before that
  (`value2`), so one record gives T for either cadence.
- **What the loop's records count**, chosen here: a row for each frame the
  gate read, a count for the frames its cadence skipped, and the frames
  missed before either (`presentedFrames`); still-to-boxes from the first
  frame under T before a hold to the end of the first detection started in
  it that wasn't stale or failed, a hold under way at the record's start
  left out. `next()` starts a detection when one ends, so `frames.started`
  counts only the starts on a frame.
- **A pinned Motion record is named by its run-relative path** in the
  config, which goes to the phone: the laptop's folders stay out of it.
- **Commits regrouped**: the scheduler pairs boxes with the tracker's
  match (`alike`, `FrameLoop`), so "gate and scheduling" can't come before
  "tracker". The gate, the Motion page and the bench make the first
  commit; scheduling, the tracker, the overlay and the live records the
  second.

The loop, as built:

- `stillness.ts`: the 64 px grey copy (luma 0.299, 0.587, 0.114) minus its
  mean, and the motion value; `StillnessGate`: still after 150 ms under T
  (compared strictly), moving at one frame over T (boxes dim), boxes
  cleared 400 ms after the first frame over T since the gate was last
  still (dips under T shorter than 150 ms don't reset it); each clearing
  is counted, and a detection started before one is stale. With no T
  (rule G off) it is always still and never rests.
- `scheduler.ts`: one detection at a time, frames dropped and never
  queued, 200 ms between starts, the rest (Decision 16); `FrameLoop` puts
  the gate, the scheduler and the tracker together, and its `next()`
  starts the next detection the moment one ends, if the picture is still.
- `tracker.ts` as planned: greedy IoU ≥ 0.3 within a class, highest first,
  ties by the shown box's order then the new box's; 60% of the way; kept
  through one miss; ids never reused.
- `useStillness.ts` reads each new frame by `requestVideoFrameCallback`
  (`requestAnimationFrame` without it) and times its own cost from drawing
  to the value; `useDetections.ts` drives the loop. Hidden, the loop stops;
  visible again, it starts over (in-flight results stale).
- `live.py`: `rule_g` and `gate()`, rule G on the pinned record at the
  gate's cadence, served in the API's config and printed at its start;
  `GATE_MOTION = "20261125-142850-m2c-app/motion-003.json"`,
  `GATE_EVERY = 2`: **T 4.42**.
- Live's **Save numbers** is gone: **Record** posts a `live` record (a scan
  of 60 s with a cue every 12 s, or a soak of 300 s; `rec=scan|soak` in the
  link): every frame and detection as columns, the config, the loop's
  constants and a summary.
- **Tests**: Vitest 175, S3's 90 each written before its code (Decision
  14): the gate 23, the scheduler and `FrameLoop` 21, the tracker 19, the
  live records 14, the gate bench 9, percentiles 4; pytest 20
  (`test_live_rules.py`, `test_live_api.py`). The late fixes each came
  with a failing test first: the frames missed before a skipped frame
  (below), the pinned record's name, the rest (Decision 16).

What was found, the laptop (Chrome 158, its FaceTime camera at 1920 ×
1080; a smoke only, none of its numbers used): the gate cost 5.3 to 8.6 ms
a frame; with no gate, Live ran 4.3 detections a second, 108 ms each
(model 82).

What was found, the phone (the OnePlus 13, CPH2691 by its client hints;
Chrome 157.0.8103.78 on 2026-11-25, 158.0.8127.42 on 2026-11-26, updated
overnight; Adreno 7xx; the back camera, 1,080 × 1,920 at 30 fps):

- **The gate bench** (ms the main thread was held a read, median; reads a
  second; camera frames the page missed): a CPU canvas averaging down
  (`medium`) 64.4, 14.8, 90 of 179; a CPU canvas sampling (`low`) 63.7,
  14.7, 89 of 177; **a GPU canvas 11.5** (p90 17.4), 30.0, 4 of 184;
  `createImageBitmap` asked to shrink 10.6, 29.8, 1 of 180. Sampling cost
  what averaging did: **the cost is bringing the whole frame to the CPU,
  not the shrinking.** The GPU canvas held from round 1 to round 2 (12.1,
  10.8 ms: Chrome didn't move it back to the CPU). The bitmap's 8% less is
  under R4's 10% and would need a gate that waits on a promise, so the GPU
  canvas was kept. Still values on the propped phone, all four: medians
  0.19 to 0.27 grey levels.
- **The check: the gate costs 11.8 ms a frame (median) in the Motion
  record and 9.4 to 13.6 in the live records: over 2 ms, so it runs on
  every other frame.** In the loop it waits behind the model on the GPU:
  in the scan, with no detection running, 11.9 ms (max 24); with one, a
  median of 8.0 but 39% of reads over 30 ms, a p90 of 104 and a max of 131,
  about 126 ms into the detection, when the model runs. With WebAssembly
  (the model on the CPU) its max was 29 ms.
- **Rule G** (Motion with the GPU copy, hand-held, 600 still and 598
  moving frames): every frame, still p95 2.94 against moving p5 2.66,
  overlapping (off); **every other frame, 4.397 against 4.440: on, T
  4.42**, a margin of 1%. The overlap comes in short runs, the longest 8
  frames (266 ms) each way, against the gate's 150 ms to call it still and
  400 ms to clear. The first phone record (the CPU copy, whose 15 reads a
  second compared frames 66 ms apart) gave 4.57 against 4.72, T 4.65,
  close to every other frame's.
- **The scan** (WebGPU float32, 60 s, five packs): 151 detections, none
  stale or failed, **4.0 a second while still**; 236 ms a detection
  (letterbox **67**, tensor 11, model 154, NMS 2; the frame's bitmap 1.4);
  **holding still to boxes 468 ms** (median of 7 holds, each got boxes);
  21% of frames gated. Every detection found at least one date box (a
  count, not a score: live frames carry no labels). The record says 2
  frames missed: the recorder lost the frames missed before a skipped
  frame (fixed before the soaks). On both soaks frames seen plus missed
  come to 30 a second, so the scan missed about 233 of 1,800 (13%).
- **The soak, resting on matched boxes** (300 s on one pack): 1,152
  detections, 3.9 a second, resting 5 s; the box count changed between
  871 of 1,151 consecutive detections (7 to 12 boxes, the keyword marks
  near 0.1641 flickering), only 68 matched the one before, and 3 in a row
  happened once; date boxes alone changed count between 35%. A detection
  took 236 ms in minutes 0 and 1 and **264 from minute 2**: the letterbox
  67 to 89 ms and the tensor 11 to 15, the model only 154 to 157. The
  CPU's stages slowed by a third at once and the GPU's barely, as a phone
  cutting its CPU's speed when warm would (not proven). 1,952 frames
  missed (22%).
- **The soak, resting on the gate** (Decision 16): 291 detections,
  **resting 297 s of 299** at 0.95 a second; 240 to 243 ms a detection in
  every minute (letterbox 70 to 74); 576 frames missed (6%); the picture
  moved 3 times, boxes back in a median of 450 ms. How warm the phone felt
  wasn't reported.
- **WebAssembly** (float32, 4 threads, 60 s on three packs): it works. 36
  detections, none failed, 35 with a date box; 1,122 ms a detection
  (model 1,005, letterbox 94), 1.2 a second while still; 10 stale (the
  phone moved during a detection); 15 holds, 9 too short for a detection
  to finish; still to boxes 1.45 s; 3 frames missed. Its model ran slower
  than S2's live 617 ms and rested 817; why wasn't looked into (a phone
  warm from two soaks, or the gate's main thread beside four threads).
- **Against what we expected**: 3 to 5 detections a second on WebGPU: 4.0;
  about 1 on WebAssembly: 1.2. Still to boxes, "150 ms plus one
  detection": 468 ms, that plus the wait for one already running. 10 to
  30% slower by the fifth minute: 12% without a working rest, 1% with
  one. The gate itself wasn't expected to cost anything.

Found on the way:

- The desktop app stops its preview servers when a session ends, and each
  start of the API draws a new key: a tab holding the old one gets 401
  (`no valid X-Live-Key`). A link pasted into the same tab that differs
  only after the `#` doesn't reload the page, so the old key stays: open
  each link in a new tab.
- The laptop's Wi-Fi address changed overnight (`.163` to `.194`); the API
  prints it only at its start, so it is read again before a link is given.
- `vite preview` serves a rebuilt `dist/` without a restart.
- Record numbers count per API start (`motion-003.json` is in two run
  folders), so a pin names its folder.

How to run it: as S2's, plus the Motion page (`#k=<key>&page=motion`:
Record motion, Bench the gate) and Live's records (`&rec=scan` or
`&rec=soak`; `&backend=wasm` for WebAssembly). A new Motion record is
pinned in `live.GATE_MOTION` (and `GATE_EVERY` by its cost), then the API
restarted.

Facts that belong elsewhere too (S5, with the owner's OK): in CLAUDE.md's
lessons, Android Chrome's canvas costs (a canvas kept on the CPU brings
the whole video frame over, 64 ms for 1,080 × 1,920 on the OnePlus 13; a
GPU canvas's small read-back takes 11.5 ms but waits behind WebGPU work,
up to 160 ms), the boxes' flicker that defeats a rest on matched boxes,
and the rested soak's flat minutes; in the laptop's notes, the preview
servers stopped by the app, a new key each start, the new-tab rule and the
changing Wi-Fi address; `vite preview` serving a rebuilt `dist/` in S5's
README.

Not checked: heat itself (a page reads no temperature, and the owner
reported none) and battery; the iPhone and float16 in the loop (S4's); a
staff-like phone (Later); the hidden page on the phone (no record holds a
`hidden` event); whether the boxes were on the right dates (no labels
live); T on anyone else's hands; the gate's copy in a worker and the
letterbox on a GPU canvas (S4, Later); WebAssembly over a soak; T for
every other frame read from `value2` of a record made at every frame, not
from the loop's own reads, which drift with its stalls (close, not the
same). The full test suite wasn't run; each commit's tests were, in a
worktree with absolute `.env` paths and `npm ci`: the first 121 Vitest
and 20 pytest, the second 175 and 20, none skipped, each built.

### S4. The optimisation ladder · laptop, then the phone

**Goal:** R2's speed half, R3 and R4 applied; the app's defaults set.

- **First, S2's two WebGPU problems.** (1) The iPhone's tab dies after a
  few WebGPU runs: try O4's graph capture (the same buffers every run),
  then the native WebGPU runtime (`onnxruntime-web/webgpu`, whose
  `storageBufferCacheMode` can be `simple` or `disabled`), with Safari's
  Web Inspector over a cable for the memory; Safari itself once. (2)
  float16 is slower than float32 because JSEP runs its 10 `Split` nodes on
  the CPU: a float16 file keeping Split in float32, or the native
  runtime, before R2's speed half is read. The Android phone decides
  (Decision 15).
- **The letterbox from a camera frame** (S3): 67 ms of the phone's 236 ms
  detection, against 19 to 37 ms from photos in S2. The worker draws the
  frame on a canvas kept on the CPU, as S3's first gate did at 64 ms; a
  canvas on the GPU, as the gate's now (Decision 17), is the first thing
  to try, ahead of O3.
- The phone's speed benches: float32 and float16 on WebGPU, float32 and
  int8 on WebAssembly, W1's threads; R2 and R3. (S1's job: int8 and
  int8-pct fail R2's accuracy half, so WebAssembly runs float32 and
  int8's speed is at most a side reading.)
- O3 (`live.py` adds the uint8 input; a test checks it gives the plain
  model's outputs on random frames), O4 (graph capture with GPU buffers),
  O5 only by its rule; each benched, R4 applied in order.
- The defaults (backend, precision, steps) written into the app's config,
  and the scan run again with them.

**Commits:** `feat: m2c uint8 input inside the graph`,
`feat: m2c webgpu graph capture`, O5's only if tried,
`docs: m2c plan, S4's ladder`

### S5. Report · laptop

`score_live.py`, `m2c_report.py`, and `results/m2c_live.md` from the
records alone; the two experiments' READMEs and `summarise.py`; the
README's M2c section (numbers only: no photo, no video, no brand);
CLAUDE.md, with the owner's OK: the milestone table gains M2c, the repo
layout gains `web/`, the phone's facts go in; M3's status line says it
resumes. Each commit's tests alone, in a worktree with absolute `.env`
paths (and `npm ci` in `web/`).

**Commits:** `feat: m2c report from the run records`,
`docs: m2c results in the readme`, `docs: m2c plan done`

---

## Order and sessions

```
S1 ─► [CPU job: int8 calibration, parity] ─► R0, R1, int8's accuracy ──┐
S2 app, benches (needs S1's float32 and float16 files only) ─► S3 loop ─► S4 ladder ─► S5
```

| Session | Slices | Ends with |
|---|---|---|
| 1 | S1's laptop part, S2 | the export smoked, boxes on the phone, the first benches |
| 2 | S1's job (on campus), S3, S4 | R0 to R4 and G read, the defaults set |
| 3 | S5 | M2c's results (M3 resumed on campus days from 2026-11-23, Decision 1) |

At home: S1's laptop part, S2 and S3. Nothing in them waits for the job:
the app runs float32 until R1 and R2 say otherwise.

## Compute budget

| Work | Where | Estimate |
|---|---|---|
| the export, float16, an 8-photo calibration | laptop | seconds |
| the v1-val smoke: 3 variants × 34 photos | laptop | under a minute, in parts |
| int8 calibration, 2 methods × 200 photos | `short` job | minutes |
| parity: 7 variants × 570 photos, about 4,000 passes at 54 to 100 ms | `short` job, 24 CPUs | under 15 minutes |
| browser benches on the laptop's GPU | laptop | about a minute each |
| the phone's benches, about 10 configurations of real-val | phone | 1 to 5 minutes each |
| the soak | phone | 5 minutes |

No GPU time. Storage: about 25 MB of model files on each machine, a few
MB of records.

## Files at the end of M2c

As under [Code layout](#code-layout), plus `results/m2c_live.md`,
`experiments/m2c-export/` and `experiments/m2c-phone/` (each a README and
a `summarise.py`), and the README's section.

## Risks

- **WebGPU missing or partial on the owner's phone** (an older Android
  phone, iOS before 26, or no WebGPU inside a worker): WebAssembly, or the
  WebGPU session on the main thread (its GPU work doesn't block the page
  much). Measured, not assumed.
- **No `shader-f16`**: float32 on WebGPU.
- **Operators that fall back to the CPU under WebGPU**, with a copy each
  way: the log names them; the export's opset or the simplifier is the
  first suspect.
- **A browser that refuses the camera or the worker on a self-signed
  page**: the fallbacks under [Serving it to the phone](#serving-it-to-the-phone).
- **The browser's resize differs from OpenCV's**: S2's check, and a
  matching resize in the worker if boxes move.
- **Heat**: phones slow down after minutes. Benches start rested, the soak
  measures it, and the rest rule limits it.
- **Memory on the phone**: a 1,280 px model and a 28 MB runtime can crash
  the tab on a phone with little memory; the records say if it happens.
- **int8 may not be faster in WebAssembly** (whether its kernels get an
  int8 dot product depends on the runtime's build and the browser): R2
  decides.
- **Live accuracy can't be scored** (live frames carry no labels): the
  demo isn't a test of the detector. M2b's numbers stand for accuracy.
- **Scope**: three sessions, then Later.

## Later

- **A staff-like Android phone** (the owner's call, 2026-11-24: none can
  be borrowed now; tested if the app meets trouble on the staff's phones).
  The OnePlus 13 every Android number came from is a 2025 flagship. On a
  phone like the staff's (one of the store's, or a budget Android with
  about 4 GB of RAM) the same built app: Live with Save numbers on each
  backend, the parity queue and the rested speed queue of S2's record.
  What may differ: no WebGPU (Chrome blocks GPU drivers it doesn't trust)
  or no `shader-f16`; several times slower; the tab killed for memory, as
  on the iPhone; heat sooner. If WebGPU is missing, WebAssembly's speed
  alone decides whether live boxes are usable there.
- **M2c in Scanify**: its capture screen gets live boxes from
  `web/src/detector/`, sharing the camera stream with its barcode reader,
  the model served by its backend. Running a model trained on the
  retailer's photos inside the retailer's own app is the owner's call,
  with the retailer.
- **A public demo**: with the retailer's OK, or with M2's detector, trained
  on public data only (weaker on store packs).
- **Scan and read** (the owner's idea, 2026-11-24, for Scanify): read the
  date live, like a barcode scanner, with no shutter. The detector finds
  the box on the phone; the box is cut from the full camera frame (sharp,
  not the 1,280 input); M3's reader reads the crop; the date is parsed and
  accepted only when several frames in a row agree (say 3 of 5), then the
  staff confirm. Frames agreeing guard against a wrong read, which is
  worse than none. Scanify already runs a live barcode loop
  (`useBarcodeScanner`), so barcode and date could come from one aim. The
  reader on the phone (M3's CRNN, 4.55M parameters, or PP-OCRv6 small, both
  ONNX) or on the server from the crop alone, a few KB, not a frame. It
  needs M2c's live detector and M3's reader first, and asks M3 for a
  reader small enough for this (not the 2B VLM, live). Changing Scanify's
  flow is the owner's call with the retailer. Which date is the expiry:
  M3 S8's roles.
- **The clicked photo on Scanify's server** (the owner's question,
  2026-11-24): the detector and M3's reader on the full photo, once per
  pack, where Scanify already sends it; the live preview stays on the
  phone. Live boxes from a server were weighed and not chosen: 1,280 px
  frames as JPEG are about 100 to 300 KB, so 2 a second is about 2 to 5
  Mbit/s of upload a phone (ten phones, 20 to 50 Mbit/s from a store);
  0.2 to 0.5 s behind the picture, as on-device WebAssembly is; nothing
  without the network; a machine always on. Compute isn't the cost: 54 ms
  a frame on 24 CPUs (M2b S13). It would also drop CLAUDE.md's goal 2,
  the checker in the phone's browser.
- **M4's checker** in the same loop: the reason M2c exists.
- **Boxes that follow the camera** between detections, by the shift
  between the gate's copies.
- **The gate's copy in its own worker** (S3): its read-back waits behind
  the model on the GPU (a p90 of 104 ms during a detection) and holds the
  page meanwhile; in a worker the wait would only delay the gate's value.
  Then it might read every frame again.
- **Boxes that flicker**: 7 to 12 boxes on a still pack, the keyword marks
  near the threshold coming and going (S3's soak). If it distracts on
  screen, show a mark only once two detections in a row have it.
- **The gyroscope as the gate** (`DeviceMotionEvent`; an iPhone asks
  permission first).
- **Torch, zoom, tap to focus.**
- **`m1280-ft` on WebGPU**: better than `n` on v2, 4.7 times its CPU time.
- **The native WebGPU runtime** (`jspi`), TopK inside the graph, a
  1280 × 720 stream, compressed runtime files, an installable offline app.
- **Battery use.**
- **4:3 frames** (960 × 1,280: a wider view, 30% more work).
