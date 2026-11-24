/**
 * Bench: parity and speed benches on served photos (benchRun.ts). Its options
 * can come from the URL's fragment, `run=1` starting it at once, e.g.
 *   #page=bench&kind=parity&set=real-val&variant=fp32&backend=webgpu&run=1
 * or several benches one after another, each kind:set:variant:backend[:threads]:
 *   #page=bench&queue=parity:real-val:fp32:webgpu,speed:real-val:fp32:wasm:1&run=1
 * (limit= and checks= apply to each; rest=300 waits 300 s before each speed
 * bench, Decision 13's rest). Every bench posts its own record; a
 * failure is posted as an error record and the queue goes on. The page's
 * title says how far it is.
 */
import { useEffect, useRef, useState } from "react";
import type { ApiConfig } from "../api";
import { hashParams } from "../api";
import { postError } from "../records";
import { backendName } from "../detector/client";
import type { Backend } from "../detector/types";
import type { BenchOptions, BenchSummary } from "./benchRun";
import { runBench } from "./benchRun";

function fromHash(config: ApiConfig): BenchOptions {
  const p = hashParams();
  const num = (k: string, d: number) => (p.has(k) ? Number(p.get(k)) : d);
  return {
    kind: p.get("kind") === "speed" ? "speed" : "parity",
    set: config.sets.includes(p.get("set") ?? "") ? p.get("set")! : config.sets[0],
    variant: p.get("variant") && p.get("variant")! in config.models ? p.get("variant")! : "fp32",
    backend: p.get("backend") === "wasm" || !("gpu" in navigator) ? "wasm" : "webgpu",
    threads: num("threads", 0), limit: num("limit", 0), pixelChecks: num("checks", 5),
  };
}

/** The fragment's queue, each entry kind:set:variant:backend[:threads] over `base`. */
function queueFromHash(base: BenchOptions): BenchOptions[] {
  const q = hashParams().get("queue");
  if (!q) return [base];
  return q.split(",").filter(Boolean).map((entry) => {
    const [kind, set, variant, backend, threads] = entry.split(":");
    return {
      ...base, kind: kind === "speed" ? "speed" : "parity", set: set || base.set, variant: variant || base.variant,
      backend: backend === "wasm" ? "wasm" : "webgpu", threads: threads ? Number(threads) : base.threads,
    };
  });
}

export function Bench({ config }: { config: ApiConfig }) {
  const [o, setO] = useState<BenchOptions>(() => fromHash(config));
  const [running, setRunning] = useState(false);
  const [done, setDone] = useState<[number, number]>([0, 0]);
  const [summary, setSummary] = useState<BenchSummary | null>(null);
  const [error, setError] = useState("");
  const started = useRef(false);

  const [queued, setQueued] = useState("");

  const start = async (list: BenchOptions[]) => {
    setRunning(true);
    setError("");
    const failed: string[] = [];
    // The screen stays on while benches run: a phone that dims stops the page.
    const lock = await (navigator as Navigator & { wakeLock?: { request(t: "screen"): Promise<{ release(): Promise<void> }> } })
      .wakeLock?.request("screen").catch(() => null);
    const rest = Number(hashParams().get("rest") ?? 0);
    for (const [i, options] of list.entries()) {
      const name = `${options.kind} ${options.set} ${options.variant} ${options.backend}`;
      // Decision 13: rested before each speed bench, so heat from the last bench doesn't slow it.
      for (let left = options.kind === "speed" ? rest : 0; left > 0; left--) {
        setQueued(`${i + 1} of ${list.length}: ${name}, resting ${left} s`);
        document.title = `bench ${i + 1}/${list.length}: resting ${left} s`;
        await new Promise((r) => setTimeout(r, 1000));
      }
      setQueued(list.length > 1 ? `${i + 1} of ${list.length}: ${name}` : "");
      setO(options);
      setSummary(null);
      document.title = `bench ${i + 1}/${list.length}: running`;
      try {
        setSummary(await runBench(options, config, (n, total) => setDone([n, total]),
          options.kind === "speed" ? rest : 0));
      } catch (e) {
        failed.push(`${name}: ${String(e)}`);
        setError(failed.join("\n"));
        postError("bench", e, options);
      }
    }
    document.title = `bench: done, ${list.length - failed.length} of ${list.length} posted`;
    await lock?.release().catch(() => {});
    setRunning(false);
  };

  useEffect(() => {
    if (!started.current && hashParams().get("run") === "1") {
      started.current = true;
      const list = queueFromHash(o);
      // run=1 out of the address, so going to Live and back, or a reload, doesn't start it again
      const rest = hashParams();
      rest.delete("run");
      history.replaceState(null, "", `${location.pathname}${location.search}#${rest.toString()}`);
      void start(list);
    }
  });

  const field = "rounded bg-slate-800 px-2 py-1.5";
  const set = <K extends keyof BenchOptions>(k: K, v: BenchOptions[K]) => setO({ ...o, [k]: v });
  return (
    <div className="flex max-w-xl flex-col gap-3 text-sm">
      <div className="grid grid-cols-2 gap-2">
        <label className="flex flex-col gap-1">Bench
          <select className={field} value={o.kind} onChange={(e) => set("kind", e.target.value as BenchOptions["kind"])}>
            <option value="parity">parity (conf 0.001)</option><option value="speed">speed (5 warm-ups)</option>
          </select></label>
        <label className="flex flex-col gap-1">Set
          <select className={field} value={o.set} onChange={(e) => set("set", e.target.value)}>
            {config.sets.map((s) => <option key={s}>{s}</option>)}
          </select></label>
        <label className="flex flex-col gap-1">Precision
          <select className={field} value={o.variant} onChange={(e) => set("variant", e.target.value)}>
            {["fp32", "fp16", "fp16-head32"].filter((v) => v in config.models).map((v) => <option key={v}>{v}</option>)}
          </select></label>
        <label className="flex flex-col gap-1">Backend
          <select className={field} value={o.backend} onChange={(e) => set("backend", e.target.value as Backend)}>
            {"gpu" in navigator && <option value="webgpu">WebGPU</option>}<option value="wasm">WebAssembly</option>
          </select></label>
        <label className="flex flex-col gap-1">WebAssembly threads (0: default)
          <input className={field} type="number" min={0} max={16} value={o.threads} onChange={(e) => set("threads", Number(e.target.value))} /></label>
        <label className="flex flex-col gap-1">Photos (0: all)
          <input className={field} type="number" min={0} value={o.limit} onChange={(e) => set("limit", Number(e.target.value))} /></label>
      </div>
      <button className="rounded bg-cyan-600 px-3 py-2 font-medium text-white disabled:opacity-50" disabled={running} onClick={() => start([o])}>
        {running ? `Running… ${done[0]} / ${done[1]}` : "Run"}
      </button>
      {queued && <p className="text-slate-300">{queued}</p>}
      {error && <p className="whitespace-pre-wrap text-red-400">{error}</p>}
      {summary && (
        <div className="rounded bg-slate-800 p-3 font-mono text-xs leading-relaxed">
          <div>record: {summary.file}</div>
          <div>{summary.photos} photos · {backendName(o.backend)} · {o.variant} · threads {summary.info.threads} · ONNX Runtime {summary.info.ortVersion}</div>
          <div>median ms: letterbox {summary.medianMs.letterbox.toFixed(1)} · tensor {summary.medianMs.tensor.toFixed(1)} · run {summary.medianMs.run.toFixed(1)} · post {summary.medianMs.post.toFixed(1)} · total {summary.medianMs.total.toFixed(1)}</div>
          <div>model {(summary.info.modelBytes / 1e6).toFixed(1)} MB in {summary.info.downloadMs.toFixed(0)} ms · session {summary.info.buildMs.toFixed(0)} ms · warm-ups {summary.info.warmupMs.map((m) => m.toFixed(0)).join(", ")} ms</div>
          <div>boxes at the threshold: {summary.boxesAtThreshold}{summary.decodedEqual !== null ? ` · decoded as PIL: ${summary.decodedEqual} of ${summary.photos}` : ""}</div>
          {summary.pixelChecks.map((c, i) => (
            <div key={i}>input {i}: max {c.max} · mean {c.mean.toFixed(3)} · {(100 * c.differing).toFixed(1)}% differ · {(100 * c.over2).toFixed(2)}% by over 2</div>
          ))}
          {summary.info.gpu && <div>GPU: {summary.info.gpu.vendor} {summary.info.gpu.architecture} · shader-f16 {summary.info.gpu.shaderF16 ? "yes" : "no"}</div>}
          {summary.info.placement.map((l, i) => <div key={i} className="text-slate-400">{l}</div>)}
        </div>
      )}
    </div>
  );
}
