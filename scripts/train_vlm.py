"""M3 S7: Qwen3-VL-2B fine-tuned with LoRA, our own loop on transformers and
peft (docs/m3-date-reader.md, "Qwen3-VL-2B", decision 18).

    python scripts/train_vlm.py train --seed 0                                  # the GPU run (S9's chain)
    python scripts/train_vlm.py train --smoke --device cpu --threads 22         # S7's CPU smoke
    python scripts/train_vlm.py merge --adapter DIR --out DIR                   # the LoRA folded in, for GGUF

What supervised fine-tuning is, here: each example is the prompt with its
image, then the answer it should give ({"expiry": ..., "mfg": ...} and
<|im_end|>). The model reads it all at once, predicting every next token,
and the loss is the cross-entropy on the answer's tokens only
(vlm.model_inputs labels everything else IGNORE): it learns what to say,
not to repeat the prompt.

The recipe (RECIPE): the base model in bf16 (float32 on a CPU); LoRA r 16,
alpha 32, dropout 0.05 on the language model's attention and MLP
projections, the vision tower and its merger frozen; AdamW, lr 1e-4,
cosine to 0 after a 3% linear warmup, weight decay 0 (the plan names none),
gradients clipped at 1.0; micro-batches of 4, 4 accumulated a step;
gradient checkpointing; one epoch over the VLM set (vlm.epoch: every
example once, the store's 3 times, shuffled by the seed). Validated EVALS
times (every 20%) on the real-val and store-val photos, the model
answering greedily (vlm.answer_with); its adapter saved each time. The run
keeps its last adapter: one epoch, no checkpoint chosen.

--smoke (S7's CPU check): SMOKE's few steps of the same loop, micro-batches
of 1, the loss on PROBE examples (held apart from the steps) before and
after, and 2 photos of each validation set answered; then `merge` and the
GGUF path (m3_cpu.sbatch vlm-smoke) prove a trained adapter converts.

The adapter has seen store photos: the checkpoint folder is owner-only
(umask 077), and per-photo answers on store-val stay in it. results/runs/
gets counts only.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from bestbefore import runs
from bestbefore.config import checkpoint_dir, generated_dir
from bestbefore.reader import parse, shards, vlm

RECIPE = {"lr": 1e-4, "warmup": 0.03, "micro": 4, "accum": 4, "weight_decay": 0.0, "clip": 1.0, "epochs": 1}
EVALS = 5                       # validated every 20%
SMOKE = {"steps": 6, "micro": 1, "accum": 2, "probe": 8, "val": 2}
WORKERS = 8                     # loader workers on a GPU node


def base_dir() -> Path:
    return checkpoint_dir() / "pretrained" / "Qwen3-VL-2B-Instruct"


def set_dir() -> Path:
    return generated_dir() / "m3" / "vlm-set"


# --- Data ---------------------------------------------------------------------------------------

class Examples:
    """The set's records in a given order, as model inputs (a torch
    Dataset by duck typing). Each loader worker opens the LMDB itself."""

    def __init__(self, root: Path, order: list[int], processor):
        self.root, self.order, self.processor = root, order, processor
        self.env = None

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int) -> dict:
        if self.env is None:
            self.env = shards.open_pile(self.root)
        ex = vlm.get(self.env, self.order[i])
        return vlm.model_inputs(self.processor, ex.image(), ex.answer)

    def close(self) -> None:
        """Close the LMDB. One the main process leaves open is inherited by
        every loader worker forked after it, and LMDB then refuses the
        worker's own open ("already open in this process")."""
        if self.env is not None:
            self.env.close()
            self.env = None


def loader(dataset: Examples, micro: int, workers: int, pad_id: int):
    import torch
    from functools import partial
    return torch.utils.data.DataLoader(dataset, batch_size=micro, shuffle=False, num_workers=workers,
                                       collate_fn=partial(vlm.collate, pad_id=pad_id),
                                       persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())


# --- The model ------------------------------------------------------------------------------------

def load_model(device: str, dtype):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_dir(), dtype=dtype).to(device)
    return model, AutoProcessor.from_pretrained(base_dir())


def with_lora(model, seed: int):
    """The model with LoRA on vlm.LORA_TARGETS, every other weight frozen;
    refuses if anything else would train."""
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    model = get_peft_model(model, LoraConfig(target_modules=vlm.LORA_TARGETS, bias="none", **vlm.LORA))
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    stray = [n for n in trainable if "lora_" not in n or ".language_model.layers." not in n]
    if stray or not trainable:
        raise RuntimeError(f"weights outside the language model's LoRA would train: {stray[:5]}")
    return model


def lr_at(step: int, steps: int, lr: float, warmup: float) -> float:
    """Linear warmup over the first `warmup` share of the steps, then
    cosine down to 0 (step counts from 0)."""
    w = max(1, round(warmup * steps))
    if step < w:
        return lr * (step + 1) / w
    return lr * 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, steps - w)))


def probe_loss(model, dataset: Examples, pad_id: int) -> float:
    """The mean loss on each example of `dataset`, one at a time, in eval
    mode (no dropout): the smoke's held-apart measure."""
    import torch
    model.eval()
    losses = []
    with torch.inference_mode():
        for i in range(len(dataset)):
            batch = {k: v.to(model.device) for k, v in vlm.collate([dataset[i]], pad_id).items()}
            losses.append(model(**batch).loss.item())
    dataset.close()
    model.train()
    return float(np.mean(losses))


def validate(model, processor, limit: int | None, private: Path, step: int) -> dict:
    """real-val and store-val photos answered greedily, scored field by
    field; per-photo answers into the private folder, counts returned."""
    import torch
    model.eval()
    summary = {}
    for name in vlm.VAL_SETS:
        photos = vlm.val_photos(name)[:limit] if limit else vlm.val_photos(name)
        rows, tally = [], {}
        for photo in photos:
            said = vlm.answer_with(model, processor, vlm.load_photo(photo.image))
            scores = vlm.score_answer(photo.answer, said["answer"])
            rows.append({"key": photo.key, "truth": None if photo.answer is None else photo.answer.as_dict(),
                         "tags": photo.tags, **{k: said[k] for k in ("text", "answer", "problems", "confidence",
                                                                     "prompt_tokens", "seconds")}, "scores": scores})
            for f, s in scores.items():
                tally.setdefault(f, {}).setdefault(s, 0)
                tally[f][s] += 1
            tally.setdefault("problems", 0)
            tally["problems"] += bool(said["problems"])
        runs.write_json(private / f"val-{name}-step{step}.json", rows)
        summary[name] = {"photos": len(photos), "scored": sum(bool(r["scores"]) for r in rows), **tally,
                         "seconds_a_photo": round(float(np.mean([r["seconds"] for r in rows])), 2) if rows else None}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.train()
    return summary


# --- Training -------------------------------------------------------------------------------------

def train(args) -> None:
    import torch
    os.umask(0o077)                     # the adapter and its answers have seen store photos
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(args.threads or os.cpu_count())
    dtype = torch.float32 if device == "cpu" or args.fp32 else torch.bfloat16
    recipe = dict(RECIPE)
    if args.smoke:
        recipe.update(micro=SMOKE["micro"], accum=SMOKE["accum"])
    root = args.set or set_dir()
    env = shards.open_pile(root)
    kinds = vlm.kinds(env)
    env.close()
    order = vlm.epoch(kinds, np.random.default_rng(args.seed))
    per_step = recipe["micro"] * recipe["accum"]
    if args.smoke:
        probe_order, order = order[:SMOKE["probe"]], order[SMOKE["probe"]:SMOKE["probe"] + SMOKE["steps"] * per_step]
    steps = SMOKE["steps"] if args.smoke else math.ceil(len(order) / per_step)
    evals = sorted({round(steps * k / EVALS) for k in range(1, EVALS + 1)}) if not args.smoke else [steps]
    name = "m3-vlm-smoke" if args.smoke else f"m3-vlm-ft-s{args.seed}"
    manifest = runs.read_json(root / "manifest.json") or {}
    record = runs.start_run(name, {"recipe": recipe, "lora": vlm.LORA, "lora_targets": vlm.LORA_TARGETS,
                                   "steps": steps, "evals_at": evals, "examples_listed": len(order),
                                   "device": device, "dtype": str(dtype), "threads": torch.get_num_threads(),
                                   "set": str(root), "set_made": manifest.get("made"),
                                   "set_count": manifest.get("count"), "smoke": args.smoke,
                                   "prompt": vlm.PROMPT, "max_pixels": vlm.MAX_PIXELS,
                                   "code_sha256": runs.code_sha256(vlm, parse, sys.modules[__name__])},
                            seed=args.seed)
    out = checkpoint_dir() / "m3" / "vlm" / record.name
    out.mkdir(parents=True)

    t0 = time.time()
    model, processor = load_model(device, dtype)
    pad_id = processor.tokenizer.pad_token_id
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = with_lora(model, args.seed)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    runs.update_run(record, load_seconds=round(time.time() - t0, 1), trainable_parameters=n_train)
    print(f"{name}: {steps} steps of {per_step}, {n_train:,} trainable, on {device} in {dtype}", flush=True)

    probe = {}
    if args.smoke:
        probe_set = Examples(root, probe_order, processor)
        probe["before"] = probe_loss(model, probe_set, pad_id)
        print(f"probe loss before: {probe['before']:.4f}", flush=True)
    data = loader(Examples(root, order, processor), recipe["micro"],
                  args.workers if args.workers is not None else (2 if device == "cpu" else WORKERS), pad_id)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    model.train()
    log, evaluated, batches = [], {}, iter(data)
    seen = tokens = 0
    t_train, wait = time.time(), 0.0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, recipe["lr"], recipe["warmup"])
        step_loss = 0.0
        for _ in range(recipe["accum"]):
            t_wait = time.time()
            batch = next(batches, None)
            wait += time.time() - t_wait
            if batch is None:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                runs.update_run(record, failed={"non_finite": True, "step": step})
                sys.exit(f"step {step}: the loss isn't finite")
            (loss / recipe["accum"]).backward()
            step_loss += loss.item() / recipe["accum"]
            seen += batch["input_ids"].shape[0]
            tokens += int(batch["attention_mask"].sum())
        norm = torch.nn.utils.clip_grad_norm_(params, recipe["clip"]).item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        log.append({"step": step + 1, "loss": round(step_loss, 5), "lr": opt.param_groups[0]["lr"],
                    "grad_norm": round(norm, 4)})
        if (step + 1) % (1 if args.smoke else 20) == 0:
            took = time.time() - t_train
            print(f"step {step + 1}/{steps}: loss {np.mean([r['loss'] for r in log[-20:]]):.4f}, "
                  f"{seen / took:.2f} examples/s, {tokens / took:.0f} tokens/s, loader wait {wait / took:.0%}",
                  flush=True)
        if step + 1 in evals:
            adapter = out / ("adapter" if step + 1 == steps else f"adapter-step{step + 1}")
            model.save_pretrained(adapter)
            t_val = time.time()
            evaluated[step + 1] = validate(model, processor, SMOKE["val"] if args.smoke else None, out, step + 1)
            evaluated[step + 1]["seconds"] = round(time.time() - t_val, 1)
            print(f"validated at step {step + 1}: {evaluated[step + 1]}", flush=True)
            runs.update_run(record, log=log, evals=evaluated)
    took = time.time() - t_train
    if args.smoke:
        data.dataset.close()            # held by this process only when the loader has no workers
        probe["after"] = probe_loss(model, Examples(root, probe_order, processor), pad_id)
        print(f"probe loss after: {probe['after']:.4f}", flush=True)
    memory = {"cuda_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)} if device == "cuda" else {}
    runs.update_run(record, log=log, evals=evaluated, probe=probe, checkpoint=str(out), adapter=str(out / "adapter"),
                    examples_seen=seen, tokens_seen=tokens, train_seconds=round(took, 1),
                    examples_a_second=round(seen / took, 3), tokens_a_second=round(tokens / took, 1),
                    loader_wait=round(wait / took, 3), **memory, ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print(f"{name}: {seen} examples in {took:.0f} s -> {out / 'adapter'}\nrun record {record}")


# --- Merging, for the GGUF path -------------------------------------------------------------------

def merge(args) -> None:
    """The base model in float32 with the adapter folded in, saved in
    bf16 as safetensors with Qwen's processor and tokenizer files beside
    it: what convert_hf_to_gguf.py reads (scripts/vlm_gguf.sh)."""
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    os.umask(0o077)
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(base_dir(), dtype=torch.float32)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    AutoProcessor.from_pretrained(base_dir()).save_pretrained(args.out)
    for name in ("chat_template.json", "generation_config.json"):     # as the base ships them
        if (base_dir() / name).exists():        # copyfile: the umask sets the mode, not the source's 664
            shutil.copyfile(base_dir() / name, args.out / name)
    print(f"merged {args.adapter} into the base in {time.time() - t0:.0f} s -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--device", choices=("cuda", "cpu"))
    t.add_argument("--threads", type=int, help="torch's threads on a CPU")
    t.add_argument("--workers", type=int, help="loader workers (default 8 on a GPU, 2 on a CPU)")
    t.add_argument("--fp32", action="store_true", help="train in float32 on the GPU too")
    t.add_argument("--set", type=Path, help="the VLM set (default $GENERATED_DIR/m3/vlm-set)")
    m = sub.add_parser("merge")
    m.add_argument("--adapter", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"train": train, "merge": merge}[args.command](args)


if __name__ == "__main__":
    main()
