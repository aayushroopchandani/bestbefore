/**
 * ONNX Runtime Web's session for the detector: the backend, the pinned
 * shape and the warm-up (docs/m2c-live-boxes.md, "Where it runs").
 *
 * onnxruntime-web 1.31.1's default import is its JSEP build: WebGPU and
 * WebAssembly in one runtime (ort-wasm-simd-threaded.jsep.wasm, 28.2 MiB).
 * Its `onnxruntime-web/webgpu` import is a different runtime since then
 * (the native WebGPU provider, an asyncify .wasm), not used here.
 *
 * No React and no app state: this folder is the part Scanify can take.
 */
import * as ort from "onnxruntime-web";
import type { DetectorConfig, GpuInfo } from "./types";

/** ONNX Runtime's WebAssembly threads: the asked number, or its own
 *  default (half the cores, 4 at most); always 1 without cross-origin
 *  isolation, which threads need (SharedArrayBuffer). */
export function wasmThreads(asked: number | undefined, cores: number, isolated: boolean): number {
  if (!isolated) return 1;
  return asked && asked > 0 ? asked : Math.min(4, Math.ceil(cores / 2));
}

/** The lines of ONNX Runtime's build log that say where nodes run. */
export function placementLines(lines: string[]): string[] {
  const out: string[] = [];
  let inList = false;
  for (const line of lines) {
    if (/placed on \[|not assigned to the preferred|Node placements/.test(line)) {
      out.push(line.trim());
      inList = /Node\(s\) placed on/.test(line);
    } else if (inList && /^\s*\S.*\(.+\)\s*$/.test(line.replace(/^\[[^\]]*\]\s*/, ""))) {
      out.push(line.trim());
    } else {
      inList = false;
    }
  }
  return out;
}

interface Built {
  session: ort.InferenceSession;
  buildMs: number;
  threads: number;
  gpu: GpuInfo | null;
  log: string[];
}

/** A session for `model`, every free dimension pinned (batch 1, the input's
 *  height and width), so ONNX Runtime optimises for fixed shapes. With
 *  `logPlacement`, its verbose log is caught while it builds. */
export async function buildSession(model: Uint8Array, config: DetectorConfig, cores: number,
  isolated: boolean): Promise<Built> {
  ort.env.wasm.wasmPaths = config.wasmPaths;
  ort.env.wasm.proxy = false; // the worker is the proxy; ONNX Runtime's own would nest a worker in it
  const threads = wasmThreads(config.threads, cores, isolated);
  ort.env.wasm.numThreads = threads;
  ort.env.logLevel = "warning";

  const log: string[] = [];
  const restore = config.logPlacement ? catchConsole(log) : () => {};
  const options: ort.InferenceSession.SessionOptions = {
    executionProviders: [config.backend],
    graphOptimizationLevel: "all",
    freeDimensionOverrides: { batch: 1, height: config.inputHeight, width: config.inputWidth },
    ...(config.logPlacement ? { logSeverityLevel: 0 as const, logVerbosityLevel: 0 } : {}),
  };
  const t0 = performance.now();
  let session: ort.InferenceSession;
  try {
    session = await ort.InferenceSession.create(model, options);
  } finally {
    restore();
  }
  const buildMs = performance.now() - t0;
  return { session, buildMs, threads, gpu: config.backend === "webgpu" ? gpuInfo() : null, log };
}

/** Runs the session on a grey input `n` times, the first being the cold
 *  run (WebGPU compiles its shaders then). Returns each run's ms. */
export async function warmUp(session: ort.InferenceSession, width: number, height: number, n: number): Promise<number[]> {
  const x = new ort.Tensor("float32", new Float32Array(3 * width * height).fill(114 / 255), [1, 3, height, width]);
  const ms: number[] = [];
  for (let i = 0; i < n; i++) {
    const t0 = performance.now();
    const out = await session.run({ [session.inputNames[0]]: x });
    ms.push(performance.now() - t0);
    for (const v of Object.values(out)) v.dispose();
  }
  return ms;
}

export function ortVersion(): string {
  return ort.env.versions.web ?? "unknown";
}

/** The adapter ONNX Runtime's WebGPU backend chose (it sets env.webgpu.adapter). */
function gpuInfo(): GpuInfo | null {
  const adapter = ort.env.webgpu.adapter as unknown as
    | { info?: Partial<Record<"vendor" | "architecture" | "device" | "description", string>>; features?: Iterable<string> }
    | undefined;
  if (!adapter) return null;
  const features = [...(adapter.features ?? [])].sort();
  const info = adapter.info ?? {};
  return {
    vendor: info.vendor ?? "", architecture: info.architecture ?? "", device: info.device ?? "",
    description: info.description ?? "", features, shaderF16: features.includes("shader-f16"),
  };
}

/** console.log/info/warn/error also copied into `into`, until the returned
 *  function is called. ONNX Runtime's WebAssembly prints its log there. */
function catchConsole(into: string[]): () => void {
  const names = ["log", "info", "warn", "error", "debug"] as const;
  const original = names.map((n) => console[n]);
  names.forEach((n, i) => {
    console[n] = (...args: unknown[]) => {
      into.push(args.map(String).join(" "));
      original[i].apply(console, args);
    };
  });
  return () => names.forEach((n, i) => (console[n] = original[i]));
}
