"""One training loop for CTC readers (docs/m3-date-reader.md, "CRNN + CTC",
decisions 16 and 17): the CRNN now, PP-OCRv6 if S6 proves the PyTorch route.

CTC's loss: for one crop, the frames' probabilities (log-softmax) and the
target text; it sums the probability of every way of laying the text out
over the frames with blanks and repeats (every path that greedy decoding
would turn into that text) and returns minus its log. So the reader is never
told where each character is. torch.nn.CTCLoss, blank 0, in float32; a
crop whose text needs more frames than it has (its length plus one blank
between each pair of equal neighbours) can't be laid out at all: its loss
is infinite, and zero_infinity drops it (counted as `infeasible`).

A run: AdamW, warmup then cosine, gradient clipping; scored m3.EVALS times
on m3.CHECKPOINT_SUITES (pooled exact text, through the orientation step),
keeping the best checkpoint; then that checkpoint exported to ONNX and
read against PyTorch on real-val (onnx_run.PARITY_MIN).
"""
from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .. import m3
from . import crnn, data, metrics, onnx_run


@dataclass
class Config:
    reader: str = "crnn"
    arm: str = "all"
    seed: int = 0
    budget: int = m3.BUDGET["crnn"]         # crops seen
    batch: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup: float = 0.02                    # share of the steps
    clip: float = 5.0
    evals: int = m3.EVALS
    workers: int = 16
    threads: int | None = None              # torch's threads in the main process (None: its default)
    amp: bool = True                        # bf16 convolutions while training
    device: str = "cuda"
    pad: float = 0.0                        # S2's padding (m3.crop_recipe_used)
    visible_min: float | None = None        # m3.VISIBLE_MIN
    suites: tuple[str, ...] = m3.CHECKPOINT_SUITES
    eval_batch: int = 128
    log_every: int = 50
    channels: tuple[int, int, int, int] = crnn.CHANNELS     # smaller only in tests
    hidden: int = crnn.HIDDEN

    @property
    def steps(self) -> int:
        return self.budget // self.batch


def build(cfg: Config) -> torch.nn.Module:
    if cfg.reader != "crnn":
        raise ValueError(f"no reader {cfg.reader!r} yet (PP-OCRv6's comes with S6)")
    return crnn.CRNN(channels=cfg.channels, hidden=cfg.hidden)


def lr_factor(step: int, steps: int, warmup: float) -> float:
    """The learning rate's share at `step` (0-based): linear warmup to 1 over
    the warmup's steps, then a cosine down to 0 at the last step."""
    w = max(1, round(warmup * steps))
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, steps - w)))


def needed_frames(classes: list[int]) -> int:
    """The fewest frames CTC can lay a text out on: one a character, and a
    blank between each pair of equal neighbours."""
    return len(classes) + sum(a == b for a, b in zip(classes, classes[1:]))


def ctc_loss(model: torch.nn.Module, batch: dict, device: str, loss_fn) -> tuple[torch.Tensor, int]:
    """The batch's mean CTC loss (a crop's loss over its text's length), and
    how many crops couldn't be laid out on their own frames."""
    x = batch["x"].to(device, non_blocking=True)
    logits = model(x)                                            # [N, T, C]
    log_probs = logits.float().log_softmax(-1).permute(1, 0, 2)  # [T, N, C], as CTCLoss takes them
    frames = onnx_run.own_frames(logits.shape[1], batch["widths"].tolist(), x.shape[3])
    lengths = batch["lengths"].tolist()
    targets = batch["targets"].tolist()
    starts = np.cumsum([0] + lengths)
    infeasible = sum(needed_frames(targets[s:s + n]) > f for s, n, f in zip(starts, lengths, frames))
    loss = loss_fn(log_probs, batch["targets"].to(device), torch.tensor(frames, device=device),
                   batch["lengths"].to(device))
    return loss, infeasible


class NonFinite(RuntimeError):
    """A step's loss or gradients weren't finite: the run stops before the
    update would spoil the weights. `args[0]` holds diagnose()'s report."""


def _replay(model: torch.nn.Module, batch: dict, device: str, amp: bool, onednn: bool) -> dict:
    """The same step again from the same weights (no update): its loss and
    the layers whose gradients aren't finite, with bf16 on or off and
    oneDNN (PyTorch's Intel kernels for convolutions and LSTMs) on or off."""
    was = model.amp
    model.amp = amp
    try:
        with torch.backends.mkldnn.flags(enabled=onednn):
            model.zero_grad(set_to_none=True)
            loss, _ = ctc_loss(model, batch, device, torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True))
            loss.backward()
    finally:
        model.amp = was
    return {"loss": loss.item(), "grads_not_finite": [n for n, p in model.named_parameters()
                                                     if p.grad is not None and not torch.isfinite(p.grad).all()]}


def diagnose(model: torch.nn.Module, batch: dict, device: str, step: int, loss: float, norm: float,
             names: list[str], keep: Path | None = None) -> dict:
    """What a non-finite step held. First, from the step itself, the layers
    whose gradients aren't finite (a NaN flows down the network, so the
    highest of them is where it began). Then the same step replayed from the
    same weights: twice as run (is it reproducible?), in float32, and with
    oneDNN off. The batch and weights go to `keep` for study offline. Last,
    crop by crop: pile and record number, width, text length, own frames,
    whether its input and logits are finite, and its own CTC loss (no
    infinities zeroed); the crops at fault are those whose input, logits or
    loss isn't finite, or, when every loss is finite, whose own loss alone
    gives non-finite gradients."""
    layers = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if keep is not None:
        torch.save({"step": step, "batch": {k: v.cpu() for k, v in batch.items()}, "model": model.state_dict(),
                    "amp": model.amp}, keep)
    replays = {"as_run_1": _replay(model, batch, device, model.amp, True),
               "as_run_2": _replay(model, batch, device, model.amp, True),
               "float32": _replay(model, batch, device, False, True),
               "no_onednn": _replay(model, batch, device, model.amp, False)}
    x = batch["x"].to(device)
    with torch.no_grad():
        logits = model(x).float()
        log_probs = logits.log_softmax(-1).permute(1, 0, 2)
        frames = onnx_run.own_frames(logits.shape[1], batch["widths"].tolist(), x.shape[3])
        each = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)(
            log_probs, batch["targets"].to(device), torch.tensor(frames, device=device), batch["lengths"].to(device))
    records = batch["records"].tolist() if "records" in batch else [None] * len(frames)
    crops = [{"pile": names[p], "record": r, "width": w, "length": n, "frames": f,
              "input_finite": bool(torch.isfinite(x[i]).all()), "logits_finite": bool(torch.isfinite(logits[i]).all()),
              "loss": float(each[i])}
             for i, (p, r, w, n, f) in enumerate(zip(batch["piles"].tolist(), records, batch["widths"].tolist(),
                                                     batch["lengths"].tolist(), frames))]
    weights = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    at_fault = [c for c in crops if not (c["input_finite"] and c["logits_finite"]) or math.isnan(c["loss"])]
    if not at_fault and not weights:
        # the losses are finite, the gradients aren't: each crop's own loss back-propagated
        # through the same batch (BatchNorm keeps the batch's statistics), as the step did
        logits = model(x).float()
        each = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=True)(
            logits.log_softmax(-1).permute(1, 0, 2), batch["targets"].to(device),
            torch.tensor(frames, device=device), batch["lengths"].to(device))
        for i, c in enumerate(crops):
            model.zero_grad(set_to_none=True)
            each[i].backward(retain_graph=i + 1 < len(crops))
            g = math.sqrt(sum(float(p.grad.float().pow(2).sum()) for p in model.parameters() if p.grad is not None))
            if not math.isfinite(g):
                at_fault.append(c | {"own_grad_norm": g})
    return {"step": step, "loss": loss, "grad_norm": norm, "weights_not_finite": weights,
            "grads_not_finite": layers, "replays": replays, "kept": str(keep) if keep else None,
            "padded_width": int(x.shape[3]), "crops": len(crops), "at_fault": at_fault,
            "infinite_losses": sum(math.isinf(c["loss"]) for c in crops)}


def evaluate(model: torch.nn.Module, suites: dict[str, tuple[list, list[str]]], device: str, batch: int) -> dict:
    """Exact text and mean character error rate per suite, read as scoring
    reads (fp32, the orientation step); `pooled`: exact over all crops."""
    reader = onnx_run.TorchReader(model, device)
    out, right, total = {}, 0, 0
    for name, (views, texts) in suites.items():
        reads = onnx_run.read_oriented(reader, views, batch)
        exact = [metrics.exact(r.text, t) for (r, _), t in zip(reads, texts)]
        out[name] = {"exact": sum(exact) / len(exact), "cer": float(np.mean([metrics.cer(r.text, t)
                                                                              for (r, _), t in zip(reads, texts)])),
                     "n": len(exact)}
        right, total = right + sum(exact), total + len(exact)
    out["pooled"] = right / total
    return out


@dataclass
class Log:
    """What a run keeps for its record, beside run.json's config."""
    history: list[dict] = field(default_factory=list)       # one entry a log window
    evals: list[dict] = field(default_factory=list)
    drawn: Counter = field(default_factory=Counter)
    infeasible: int = 0
    crops: int = 0
    train_seconds: float = 0.0          # forward, backward, step
    wait_seconds: float = 0.0           # waiting for the loader
    eval_seconds: float = 0.0


def run(cfg: Config, root: Path, checkpoints: Path, say=print) -> dict:
    """Train one reader on its arm; the best checkpoint (and the last) into
    `checkpoints`. Returns the run's summary for its record."""
    torch.manual_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    device = cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
    mix = data.Mix.for_arm(root, cfg.arm, cfg.visible_min)
    suites = {name: data.suite(root, name, cfg.pad) for name in cfg.suites}
    stream = data.Stream(root, mix, cfg.batch, cfg.pad, cfg.seed)
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=cfg.workers, pin_memory=device == "cuda",
        persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers else None,
        worker_init_fn=data.worker_init if cfg.workers else None)
    model = build(cfg).to(device)
    model.amp = cfg.amp
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda s: lr_factor(s, cfg.steps, cfg.warmup))
    loss_fn = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    every = max(1, cfg.steps // cfg.evals)
    log, best, window = Log(), None, []
    say(f"{cfg.reader} on {cfg.arm}: {cfg.steps} steps of {cfg.batch}, {crnn.parameters(model):,} parameters, "
        f"device {device}, amp {cfg.amp}, {cfg.workers} workers, {torch.get_num_threads()} threads; "
        f"draws {mix.describe()}")
    batches = iter(loader)
    model.train()
    try:
        for step in range(cfg.steps):
            t0 = time.perf_counter()
            batch = next(batches)
            t1 = time.perf_counter()
            loss, infeasible = ctc_loss(model, batch, device, loss_fn)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip))
            value = loss.item()
            if not (math.isfinite(value) and math.isfinite(norm)):
                raise NonFinite(diagnose(model, batch, device, step + 1, value, norm, mix.names,
                                         keep=checkpoints / "non-finite-step.pt"))
            optimiser.step()
            schedule.step()
            t2 = time.perf_counter()
            log.wait_seconds += t1 - t0
            log.train_seconds += t2 - t1
            log.crops += len(batch["lengths"])
            log.infeasible += infeasible
            log.drawn.update(mix.names[p] for p in batch["piles"].tolist())
            window.append(value)
            if (step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps:
                seconds = log.train_seconds + log.wait_seconds
                entry = {"step": step + 1, "loss": float(np.mean(window)), "lr": schedule.get_last_lr()[0],
                         "crops_per_second": log.crops / seconds, "waiting": log.wait_seconds / seconds}
                log.history.append(entry)
                say(f"step {step + 1}: loss {entry['loss']:.3f}, lr {entry['lr']:.2e}, "
                    f"{entry['crops_per_second']:.0f} crops/s, {entry['waiting']:.0%} waiting for the loader")
                window = []
            if (step + 1) % every == 0 or step + 1 == cfg.steps:
                t3 = time.perf_counter()
                scores = evaluate(model, suites, device, cfg.eval_batch)
                log.eval_seconds += time.perf_counter() - t3
                scores["step"] = step + 1
                log.evals.append(scores)
                say(f"eval at step {step + 1}: exact pooled {scores['pooled']:.3f}, "
                    + ", ".join(f"{n} {scores[n]['exact']:.3f} (cer {scores[n]['cer']:.3f})" for n in cfg.suites))
                if best is None or scores["pooled"] > best["pooled"]:
                    best = scores
                    save(model, cfg, step + 1, scores, checkpoints / "best.pt")
                model.train()
        save(model, cfg, cfg.steps, log.evals[-1], checkpoints / "last.pt")
    finally:
        # the loader's last references (persistent workers keep its iterator): its workers
        # end, or its in-process stream closes its LMDB files (LMDB refuses a file already
        # open in the process, as a stopped run's would be)
        del batches, loader
    return {"steps": cfg.steps, "crops": log.crops, "infeasible": log.infeasible,
            "crops_per_second": log.crops / (log.train_seconds + log.wait_seconds),
            "crops_per_second_training": log.crops / log.train_seconds,
            "waiting_share": log.wait_seconds / (log.train_seconds + log.wait_seconds),
            "eval_seconds": round(log.eval_seconds, 1), "train_seconds": round(log.train_seconds, 1),
            "wait_seconds": round(log.wait_seconds, 1),
            "drawn": {n: log.drawn[n] for n in mix.names}, "mix": mix.describe(),
            "history": log.history, "evals": log.evals, "best": {"step": best["step"], "pooled": best["pooled"]},
            "parameters": crnn.parameters(model), "device": device}


def save(model: torch.nn.Module, cfg: Config, step: int, scores: dict, path: Path) -> None:
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step, "scores": scores}, path)


def load(path: Path) -> tuple[torch.nn.Module, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in saved["config"].items()})
    model = build(cfg)
    model.load_state_dict(saved["model"])
    return model.eval(), saved


def export_checked(checkpoint: Path, root: Path, threads: int, suite_name: str = "real-val") -> dict:
    """The checkpoint exported beside itself as ONNX (reader.onnx) and read
    against PyTorch on a suite, both turns of every crop: parity is the
    share of reads with the same text (decision 16: at least PARITY_MIN)."""
    model, saved = load(checkpoint)
    path = onnx_run.export(model, checkpoint.with_name("reader.onnx"))
    views, _ = data.suite(root, suite_name, saved["config"]["pad"])
    turned = [np.ascontiguousarray(np.rot90(v, q)) for v in views for q in (0, 2)]
    torch_reads = onnx_run.read(onnx_run.TorchReader(model), turned)
    t0 = time.perf_counter()
    onnx_reads = onnx_run.read(onnx_run.OnnxReader(path, threads), turned)
    seconds = time.perf_counter() - t0
    parity = onnx_run.parity(torch_reads, onnx_reads)
    return {"onnx": str(path), "suite": suite_name, "reads": len(turned), "parity": parity,
            "passed": parity >= onnx_run.PARITY_MIN, "onnx_ms_per_read": round(1000 * seconds / len(turned), 2),
            "threads": threads, "differing": [(a.text, b.text) for a, b in zip(torch_reads, onnx_reads)
                                              if a.text != b.text][:20]}
