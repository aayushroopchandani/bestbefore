/**
 * The main thread's typed handle on the detector worker.
 *
 *   const detector = await Detector.start(config);   // the model loaded, warmed up
 *   const result = await detector.detect(bitmap, conf);
 *
 * The bitmap is transferred to the worker (not copied) and closed there.
 * One detection at a time: a second detect() before the first ends is
 * refused, since the loop's rule is to drop frames, never queue them.
 *
 * No React and no app state: this folder is the part Scanify can take.
 */
import type { Backend, DetectorConfig, DetectorInfo, DetectResult, FromWorker, ToWorker } from "./types";

export function backendName(b: Backend): string {
  return b === "webgpu" ? "WebGPU" : "WebAssembly";
}

export class Detector {
  readonly info: DetectorInfo;
  readonly config: DetectorConfig;
  private readonly worker: Worker;
  private nextId = 1;
  private pending: { id: number; resolve: (r: DetectResult) => void; reject: (e: Error) => void } | null = null;

  private constructor(worker: Worker, info: DetectorInfo, config: DetectorConfig) {
    this.worker = worker;
    this.info = info;
    this.config = config;
    worker.onmessage = (event: MessageEvent<FromWorker>) => this.receive(event.data);
  }

  /** Starts a worker, loads the model in it and warms it up. */
  static start(config: DetectorConfig): Promise<Detector> {
    const worker = new Worker(new URL("./detector.worker.ts", import.meta.url), { type: "module", name: "detector" });
    return new Promise((resolve, reject) => {
      worker.onmessage = (event: MessageEvent<FromWorker>) => {
        const m = event.data;
        if (m.type === "ready") resolve(new Detector(worker, m.info, config));
        else if (m.type === "error") {
          worker.terminate();
          reject(new Error(m.message));
        }
      };
      worker.onerror = (e) => {
        worker.terminate();
        reject(new Error(e.message || "the detector worker failed to start"));
      };
      post(worker, { type: "init", config });
    });
  }

  private isClosed = false;

  get busy(): boolean {
    return this.pending !== null;
  }

  /** Closed by its owner: a detection it cut short isn't a failure. */
  get closed(): boolean {
    return this.isClosed;
  }

  /** Boxes over `conf` in `frame`, in frame pixels. */
  detect(frame: ImageBitmap, conf: number, returnInput = false): Promise<DetectResult> {
    if (this.pending) {
      frame.close();
      return Promise.reject(new Error("a detection is running: drop this frame"));
    }
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending = { id, resolve, reject };
      post(this.worker, { type: "detect", id, frame, conf, returnInput }, [frame]);
    });
  }

  close(): void {
    this.isClosed = true;
    this.worker.terminate();
    this.pending?.reject(new Error("the detector was closed"));
    this.pending = null;
  }

  private receive(m: FromWorker): void {
    const p = this.pending;
    if (!p || m.type === "ready") return;
    if (m.type === "result" && m.result.id === p.id) {
      this.pending = null;
      p.resolve(m.result);
    } else if (m.type === "error" && (m.id === undefined || m.id === p.id)) {
      this.pending = null;
      p.reject(new Error(m.message));
    }
  }
}

function post(worker: Worker, message: ToWorker, transfer: Transferable[] = []): void {
  worker.postMessage(message, transfer);
}
