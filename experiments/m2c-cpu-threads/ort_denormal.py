"""M2c S1 diagnosis, second part: ONNX Runtime alone is fast on random input
(52 ms, job 74150) but slow inside the parity job on photos (283 ms). Are
the photos' activations denormal floats? Times session.run on real-val's
first photos (ExpDate, public), letterboxed square and tight, on a flat
grey input and on noise, with ONNX Runtime's flush-to-zero off and on, and
counts how many of the output tensor's values are denormal. Prints JSON lines.

    python ort_denormal.py MODEL.onnx
"""
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

from bestbefore import live, sets
from bestbefore.images import load_upright

model = sys.argv[1]
cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))


def session(ftz: bool):
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = cpus, 1
    if ftz:
        o.add_session_config_entry("session.set_denormal_as_zero", "1")
    return ort.InferenceSession(model, o, providers=["CPUExecutionProvider"])


def ms(s, x, n=7):
    s.run(None, {"images": x})
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        s.run(None, {"images": x})
        ts.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(ts)), 1)


inputs = {"noise": np.random.default_rng(0).random((1, 3, 1280, 1280), dtype=np.float32),
          "grey 114": np.full((1, 3, 1280, 1280), 114 / 255, np.float32)}
for i, s in enumerate(sets.load_set("real-val")[:4]):
    image = load_upright(s.image)
    inputs[f"real-val {i} square"] = live.letterboxed(image, tight=False)
    inputs[f"real-val {i} tight"] = live.letterboxed(image, tight=True)
sessions = {False: session(False), True: session(True)}
tiny = np.finfo(np.float32).tiny
for name, x in inputs.items():
    out = sessions[False].run(None, {"images": x})[0]
    same = np.abs(out - sessions[True].run(None, {"images": x})[0]).max()
    print(json.dumps({"input": name, "shape": list(x.shape[2:]), "ms": ms(sessions[False], x),
                      "ms_flush_to_zero": ms(sessions[True], x), "max_output_diff": float(same),
                      "output_denormals": int(((out != 0) & (np.abs(out) < tiny)).sum())}), flush=True)
