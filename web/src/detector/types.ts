/**
 * What the main thread and the detector worker say to each other.
 * No React and no app state: this folder is the part Scanify can take.
 */

export type Backend = "webgpu" | "wasm";

export interface DetectorConfig {
  modelUrl: string; // fetched by the worker
  headers?: Record<string, string>; // sent with it (the API's key)
  backend: Backend;
  wasmPaths: string; // where ONNX Runtime's .mjs and .wasm are served, e.g. "/ort/"
  inputWidth: number; // the shape every session run gets, pinned (freeDimensionOverrides)
  inputHeight: number;
  tight?: boolean; // pad tight, not to the square (R1 kept square: false)
  threads?: number; // WebAssembly threads; 0 or none: ONNX Runtime's default
  warmups?: number; // runs on a grey input before the first detection (default 1)
  logPlacement?: boolean; // ONNX Runtime's verbose log while the session is built
}

/** What the worker found while starting: for every record. */
export interface DetectorInfo {
  ortVersion: string;
  backend: Backend;
  crossOriginIsolated: boolean;
  threads: number; // ONNX Runtime's WebAssembly threads
  modelBytes: number;
  downloadMs: number;
  buildMs: number; // InferenceSession.create
  warmupMs: number[]; // each warm-up run; the first is the cold one
  input: { name: string; dims: number[] };
  output: { name: string; dims: number[] };
  gpu: GpuInfo | null; // WebGPU's adapter, when the backend is WebGPU
  placement: string[]; // ONNX Runtime's log lines naming where nodes run
  logLines: number; // how many log lines the build printed in all
}

export interface GpuInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  features: string[];
  shaderF16: boolean;
}

/** One box in frame pixels. */
export interface FrameBox {
  cls: number;
  conf: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Milliseconds a detection spent in each stage, in the worker. */
export interface StageMs {
  letterbox: number; // drawing the frame into the input canvas, reading its pixels back
  tensor: number; // bytes to floats, channels first
  run: number; // ONNX Runtime: upload, the model, the output read back
  post: number; // best class, threshold, NMS, boxes back to the frame
  total: number;
}

export interface DetectResult {
  id: number;
  frameWidth: number;
  frameHeight: number;
  inputWidth: number;
  inputHeight: number;
  boxes: FrameBox[];
  ms: StageMs;
  input?: Uint8ClampedArray; // the input's RGBA bytes, when asked for (the bench's pixel check)
}

export type ToWorker =
  | { type: "init"; config: DetectorConfig }
  | { type: "detect"; id: number; frame: ImageBitmap; conf: number; returnInput?: boolean };

export type FromWorker =
  | { type: "ready"; info: DetectorInfo }
  | { type: "result"; result: DetectResult }
  | { type: "error"; id?: number; message: string };
