"""M2c S1 diagnosis, third part: ONNX Runtime's session alone runs a 1,280
square in about 50 to 150 ms on 24 CPUs of cn1 (jobs 74150, 74151), but
Ultralytics' predictions with the same file took 283 ms a call in the parity
job (74149). Replays the job's path, Ultralytics' ONNX backend with every
session on the job's CPUs (live.ort_threads), on real-val's first 12 photos
square, with torch at Ultralytics' threads, then 24, then 1; reports the
call's median ms and Ultralytics' own split (preprocess, inference,
postprocess). Prints JSON lines.

    python ort_ultralytics.py MODEL.onnx
"""
import json
import os
import sys
import time

os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("YOLO_OFFLINE", "1")
import numpy as np
import torch
from ultralytics import YOLO

from bestbefore import live, sets
from bestbefore.images import load_upright

cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
images = [load_upright(s.image) for s in sets.load_set("real-val")[:12]]
with live.ort_threads(cpus) as made:
    model = YOLO(sys.argv[1], task="detect")
    model.predict(source=[images[0]], imgsz=1280, conf=0.001, device="cpu", rect=False, verbose=False)
session = model.predictor.model.backend.session.get_session_options().intra_op_num_threads
for torch_threads in (None, cpus, 1):
    if torch_threads:
        torch.set_num_threads(torch_threads)
    calls, split = [], []
    for image in images:
        t = time.perf_counter()
        r = model.predict(source=[image], imgsz=1280, conf=0.001, device="cpu", rect=False, verbose=False)[0]
        calls.append((time.perf_counter() - t) * 1000)
        split.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
    med = np.median(np.array(split), axis=0).round(1).tolist()
    print(json.dumps({"OMP_WAIT_POLICY": os.environ.get("OMP_WAIT_POLICY"), "ort_threads": session,
                      "sessions_made": made, "torch_threads": torch.get_num_threads(),
                      "call_ms": round(float(np.median(calls)), 1),
                      "preprocess_inference_postprocess_ms": med}), flush=True)
