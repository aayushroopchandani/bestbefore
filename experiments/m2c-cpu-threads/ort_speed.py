"""M2c S1 diagnosis (job 74149's ONNX passes took 283 ms against PyTorch's 33
on cn1): ONNX Runtime's speed on a fixed 1,280 square input, alone, after
PyTorch has run in the same process, and with its threads spinning or not.
One process per mode, so nothing carries over. Prints one JSON line a step.

    python ort_speed.py MODEL.onnx PT_WEIGHTS MODE      (MODE: alone | after-torch)
"""
import json
import os
import sys
import time

os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ.setdefault("YOLO_OFFLINE", "1")
import numpy as np

model, weights, mode = sys.argv[1], sys.argv[2], sys.argv[3]
cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
x = np.random.default_rng(0).random((1, 3, 1280, 1280), dtype=np.float32)


def say(step, **fields):
    print(json.dumps({"mode": mode, "step": step, "affinity": len(os.sched_getaffinity(0)), **fields}), flush=True)


def ort_ms(threads, spinning=True, n=15):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    if not spinning:
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    s = ort.InferenceSession(model, o, providers=["CPUExecutionProvider"])
    s.run(None, {"images": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        s.run(None, {"images": x})
        ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 1)


say("start", cpus=cpus, omp=os.environ.get("OMP_NUM_THREADS"))
if mode == "alone":
    for threads in (cpus, cpus // 2, 1):
        say("ort", threads=threads, ms=ort_ms(threads))
    say("ort", threads=cpus, spinning=False, ms=ort_ms(cpus, spinning=False))
else:
    import torch
    from PIL import Image
    from ultralytics import YOLO
    image = Image.fromarray((x[0].transpose(1, 2, 0) * 255).astype(np.uint8))
    yolo = YOLO(weights)
    yolo.predict(source=[image], imgsz=1280, conf=0.001, device="cpu", rect=False, verbose=False)
    say("ultralytics set torch", torch_threads=torch.get_num_threads())
    for torch_threads in (cpus, 8, 1):
        torch.set_num_threads(torch_threads)
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            yolo.predict(source=[image], imgsz=1280, conf=0.001, device="cpu", rect=False, verbose=False)
            ts.append((time.perf_counter() - t) * 1000)
        say("pytorch predict", torch_threads=torch_threads, ms=round(float(np.median(ts)), 1))
        say("ort after it", torch_threads=torch_threads, threads=cpus, ms=ort_ms(cpus))
        say("ort after it", torch_threads=torch_threads, threads=cpus, spinning=False, ms=ort_ms(cpus, spinning=False))
