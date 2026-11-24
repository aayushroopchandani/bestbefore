/**
 * The detector's Web Worker (docs/m2c-live-boxes.md, "One detection"): the
 * main thread keeps the camera preview smooth while this runs the model.
 *
 *   init    fetch the model, build the session (pinned shape), warm up
 *   detect  a frame (an ImageBitmap, transferred) to boxes in frame pixels:
 *           letterbox into an OffscreenCanvas, bytes to floats, the model,
 *           best class over conf, NMS, boxes back to the frame; each
 *           stage's ms. One detection at a time: the caller waits for each.
 *
 * No React and no app state: this folder is the part Scanify can take.
 */
import type { InferenceSession } from "onnxruntime-web";
import { Tensor } from "onnxruntime-web";
import { drawLetterboxed, letterboxParams, scaleBoxes, toTensor } from "./letterbox";
import { postprocess } from "./postprocess";
import { buildSession, ortVersion, placementLines, warmUp } from "./session";
import type { DetectorConfig, DetectResult, FromWorker, StageMs, ToWorker } from "./types";

let config: DetectorConfig | null = null;
let session: InferenceSession | null = null;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let input: Float32Array<ArrayBuffer> | null = null;

function send(message: FromWorker, transfer: Transferable[] = []): void {
  self.postMessage(message, transfer);
}

async function init(c: DetectorConfig): Promise<void> {
  config = c;
  const t0 = performance.now();
  const response = await fetch(c.modelUrl, { headers: c.headers });
  if (!response.ok) throw new Error(`the model: HTTP ${response.status}`);
  const model = new Uint8Array(await response.arrayBuffer());
  const downloadMs = performance.now() - t0;

  const built = await buildSession(model, c, navigator.hardwareConcurrency || 1, self.crossOriginIsolated);
  session = built.session;
  const canvas = new OffscreenCanvas(c.inputWidth, c.inputHeight);
  // willReadFrequently: a canvas kept in memory, not on the GPU, since every
  // frame's pixels are read back.
  ctx = canvas.getContext("2d", { willReadFrequently: true, alpha: false });
  if (!ctx) throw new Error("no 2D context for an OffscreenCanvas");
  input = new Float32Array(3 * c.inputWidth * c.inputHeight);
  const warmupMs = await warmUp(session, c.inputWidth, c.inputHeight, c.warmups ?? 1);

  const meta = (name: string, list: readonly InferenceSession.ValueMetadata[]) => {
    const m = list.find((v) => v.name === name);
    return { name, dims: m && m.isTensor ? m.shape.map((d) => (typeof d === "number" ? d : -1)) : [] };
  };
  send({
    type: "ready",
    info: {
      ortVersion: ortVersion(), backend: c.backend, crossOriginIsolated: self.crossOriginIsolated,
      threads: built.threads, modelBytes: model.byteLength, downloadMs, buildMs: built.buildMs, warmupMs,
      input: meta(session.inputNames[0], session.inputMetadata),
      output: meta(session.outputNames[0], session.outputMetadata),
      gpu: built.gpu, placement: placementLines(built.log), logLines: built.log.length,
    },
  });
}

async function detect(id: number, frame: ImageBitmap, conf: number, returnInput: boolean): Promise<void> {
  if (!config || !session || !ctx || !input) throw new Error("detect before init");
  const t0 = performance.now();
  const lb = letterboxParams(frame.width, frame.height, config.tight ?? false);
  if (lb.inputWidth !== config.inputWidth || lb.inputHeight !== config.inputHeight) {
    frame.close();
    throw new Error(`a ${frame.width} x ${frame.height} frame needs a ${lb.inputWidth} x ${lb.inputHeight} input; ` +
      `the session is pinned at ${config.inputWidth} x ${config.inputHeight}`);
  }
  drawLetterboxed(ctx, frame, lb);
  frame.close();
  const pixels = ctx.getImageData(0, 0, lb.inputWidth, lb.inputHeight).data;
  const t1 = performance.now();
  toTensor(pixels, lb.inputWidth, lb.inputHeight, input);
  const t2 = performance.now();

  const x = new Tensor("float32", input, [1, 3, lb.inputHeight, lb.inputWidth]);
  const outputs = await session.run({ [session.inputNames[0]]: x });
  const y = outputs[session.outputNames[0]];
  const data = (await y.getData()) as Float32Array;
  const [, channels, anchors] = y.dims;
  const t3 = performance.now();

  const kept = postprocess(data, anchors, channels - 4, { conf });
  for (const v of Object.values(outputs)) v.dispose();
  const mapped = scaleBoxes(kept.map((b) => [b.x1, b.y1, b.x2, b.y2]), lb.inputWidth, lb.inputHeight,
    lb.frameWidth, lb.frameHeight);
  const boxes = kept.map((b, i) => {
    const [x1, y1, x2, y2] = mapped[i];
    return { cls: b.cls, conf: b.conf, x1, y1, x2, y2 };
  });
  const t4 = performance.now();

  const ms: StageMs = { letterbox: t1 - t0, tensor: t2 - t1, run: t3 - t2, post: t4 - t3, total: t4 - t0 };
  const result: DetectResult = {
    id, frameWidth: lb.frameWidth, frameHeight: lb.frameHeight, inputWidth: lb.inputWidth,
    inputHeight: lb.inputHeight, boxes, ms, ...(returnInput ? { input: pixels } : {}),
  };
  send({ type: "result", result }, returnInput ? [pixels.buffer] : []);
}

self.onmessage = (event: MessageEvent<ToWorker>) => {
  const m = event.data;
  const run = m.type === "init" ? init(m.config) : detect(m.id, m.frame, m.conf, m.returnInput ?? false);
  run.catch((e: unknown) => send({ type: "error", ...(m.type === "detect" ? { id: m.id } : {}), message: String(e) }));
};
