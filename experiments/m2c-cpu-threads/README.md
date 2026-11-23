# M2c / CPU threads: why were the ONNX passes slow on the cluster?

## Question

In S1's parity job (74149, 24 CPUs of `cn1`), Ultralytics' predictions
with the exported detector took 283 ms a photo through ONNX Runtime
(`det-fp32.onnx`, square), against 33 ms through PyTorch with the same
weights. On the laptop it was the other way round (134 against 166 ms).
Where does the time go, and how is it avoided?

## Why it matters

No rule reads these times: S1's rules read accuracy, and R2's speed half
is the phone's (S4). But the parity records keep "ms a pass", and a report
built from them would say ONNX Runtime is eight times slower than PyTorch
on a server CPU, which is untrue. Any later job that runs an ONNX model
through Ultralytics on the cluster would pay the same.

## Method

Three scripts, each run in its own `short` job of 24 CPUs on `cn1`
(2 × 24 cores, one thread a core), while job 74149 used the node's other
24 CPUs, with the main venv and `OMP_NUM_THREADS=24` as in job 74149:

| Script | Job | What it times |
|---|---|---|
| `ort_speed.py` | 74150 | `session.run` alone on a fixed 1,280 square of noise: torch never imported (24, 12 and 1 threads; spinning off), then after PyTorch predictions in the same process (torch at 24, 8 and 1 threads) |
| `ort_denormal.py` | 74151 | `session.run` on real-val's first 4 photos (ExpDate, public), square and tight, a flat grey square and noise, with ONNX Runtime's flush-to-zero (`session.set_denormal_as_zero`) off and on |
| `ort_ultralytics.py` | 74152 | the parity job's own path: Ultralytics' ONNX backend (`live.ort_threads(24)`) on real-val's first 12 photos, square, with torch at Ultralytics' threads, 24, then 1; Ultralytics' own split (`r.speed`); run twice, once with `OMP_WAIT_POLICY=PASSIVE` |

Each was submitted from `~/bestbefore` (the scripts copied to an
owner-only folder on the cluster) as:

```bash
sbatch --partition=short --cpus-per-task=24 --mem=16G --time=00:20:00 --job-name=m2c-ortdiag \
    --output=slurm/logs/%x-%j.out --wrap "cd ~/bestbefore && module load Python/3.11.11 && \
    source .venv/bin/activate && export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK YOLO_OFFLINE=1 \
    YOLO_AUTOINSTALL=false && python DIR/ort_ultralytics.py MODELS/det-fp32.onnx"
```

`ort_speed.py` takes the model, M2b's `last.pt` and a mode (`alone`,
`after-torch`); `ort_denormal.py` and `ort_ultralytics.py` take the model.
Each prints one JSON line a step; the numbers below are those lines.

## Result

**ONNX Runtime itself is fast.** Alone, on noise: 52.5 ms on 24 threads,
60.1 on 12, 268 on 1; 56 with spinning off. After PyTorch predictions in
the same process, at torch's 24, 8 or 1 threads: 51.4 to 51.6 ms. So torch
having run first doesn't slow it.

**Not denormal floats.** On real photos, square and tight, flush-to-zero
changed no output (largest difference 0) and gave no steady speed-up
(56 to 149 ms off, 52 to 161 on, in no consistent direction: the node was
shared with job 74149).

**The time is torch's, around ONNX Runtime**, in Ultralytics' own split:

| `OMP_WAIT_POLICY` | torch threads | preprocess | inference | postprocess | a call |
|---|---|---|---|---|---|
| unset (libgomp's default) | 8 (Ultralytics' own) | 101.8 | 55.8 | 159.7 | 328.2 |
| unset | 24 (as job 74149 set) | 166.2 | 63.1 | 256.9 | 490.7 |
| unset | 1 | 15.6 | 51.3 | 0.9 | 70.7 |
| `PASSIVE` | 8 | 16.9 | 51.5 | 1.1 | 72.4 |
| `PASSIVE` | 24 | 16.2 | 52.7 | 1.5 | 72.4 |
| `PASSIVE` | 1 | 15.9 | 52.7 | 0.9 | 71.9 |

(ms, medians of 12 photos.) Ultralytics turns each photo into the model's
input and decodes its output (NMS) with torch. torch's wheel bundles
GNU's OpenMP (libgomp), whose threads spin for a while after each
parallel step; beside ONNX Runtime's own spinning thread pool on the same
24 CPUs, torch's small steps crawl. With OpenMP's threads sleeping
(`PASSIVE`), or torch on one thread, the call is 72 ms: 52 of inference
and 20 of Ultralytics' own work. Ultralytics set torch to 8 threads at
its first prediction, as its rule says (min(8, CPUs − 1) of the node's 48).

## Decision

- `slurm/m2c_cpu.sbatch` exports `OMP_WAIT_POLICY=PASSIVE` (S1's fix
  commit), so a rerun times ONNX Runtime as it is. Job 74149's records
  were made before it: their boxes and scores stand (the policy changes
  when threads sleep, not what they compute), but their ONNX times are
  about 4 times too long (float32 square 281 to 301 ms a call against 72)
  and are not quoted. PyTorch's (33 to 42 ms) are right.
- CLAUDE.md's HPC section says it, for any later job running an ONNX
  model through Ultralytics, or ONNX Runtime beside torch, on the cluster.
