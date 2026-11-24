/**
 * The benches (docs/m2c-live-boxes.md, "The app"; S2). Both feed the
 * detector served photos through the same worker as Live, its session
 * pinned at 1,280 x 1,280: R1 kept square padding, so every photo and every
 * camera frame reaches the model at that shape.
 *
 * parity  every photo at conf 0.001, its boxes and stage times posted for
 *         scoring against the cluster job's Ultralytics records
 *         (scripts/score_live.py). Two pixel checks say where a moved box
 *         comes from: the photo as the browser decodes it against PIL's
 *         decode (by SHA-256, every photo), and the model's input against
 *         Ultralytics' LetterBox, value by value (the first few photos).
 * speed   5 warm-ups, then every photo at M2b's threshold, as live; each
 *         stage timed. Speed doesn't depend on what's in the picture, so
 *         photos stand in for frames and every configuration sees the same.
 */
import type { ApiConfig } from "../api";
import { apiFetch, apiJson, keyHeaders } from "../api";
import { Detector } from "../detector/client";
import type { Backend, DetectorInfo, StageMs } from "../detector/types";
import { APP, deviceInfo, postRecord } from "../records";

export const PARITY_CONF = 0.001; // every box kept, as M2b's scoring and the job kept them
const SQUARE = 1280;

export interface BenchOptions {
  kind: "parity" | "speed";
  set: string;
  variant: string;
  backend: Backend;
  threads: number; // WebAssembly threads, 0: ONNX Runtime's default
  limit: number; // the set's first N photos, 0: all
  pixelChecks: number; // parity: the first N photos' inputs checked against Ultralytics'
}

export interface PixelCheck {
  max: number; // the largest difference of one value (0 to 255)
  mean: number;
  differing: number; // the share of values that differ at all
  over2: number; // the share that differ by more than 2
}

export interface BenchSummary {
  file: string;
  photos: number;
  medianMs: StageMs;
  boxesAtThreshold: number;
  decodedEqual: number | null;
  pixelChecks: PixelCheck[];
  info: DetectorInfo;
}

interface PhotoRecord {
  index: number;
  size: [number, number];
  input: number[];
  ms: StageMs;
  boxes: number[][]; // [class, x1, y1, x2, y2, conf]: xyxy 0 to 1 of the photo, as the job's records
  decoded_equal?: boolean;
  pixel_check?: PixelCheck;
}

const r5 = (v: number) => Math.round(v * 1e5) / 1e5;

export async function runBench(o: BenchOptions, config: ApiConfig,
  progress: (done: number, total: number) => void, restSeconds = 0): Promise<BenchSummary> {
  const started = new Date().toISOString();
  const device = await deviceInfo();
  const listing = await apiJson<{ photos: { index: number; width: number; height: number }[] }>(`/api/bench/${o.set}`);
  const photos = o.limit > 0 ? listing.photos.slice(0, o.limit) : listing.photos;
  const model = config.models[o.variant];
  const conf = o.kind === "parity" ? PARITY_CONF : config.threshold;
  const detector = await Detector.start({
    modelUrl: model.url, headers: keyHeaders(), backend: o.backend, wasmPaths: "/ort/",
    inputWidth: SQUARE, inputHeight: SQUARE, threads: o.threads, warmups: o.kind === "speed" ? 5 : 1,
    logPlacement: true,
  });
  const records: PhotoRecord[] = [];
  try {
    for (const [n, p] of photos.entries()) {
      // Never from the browser's cache: each photo's request in the API's log
      // is then the bench's progress, the one trace left if the tab crashes.
      const response = await apiFetch(`/api/bench/${o.set}/${p.index}`, { cache: "no-store" });
      const decodedSha = response.headers.get("X-Decoded-Sha256");
      const frame = await createImageBitmap(await response.blob(),
        { colorSpaceConversion: "none", premultiplyAlpha: "none", imageOrientation: "from-image" });
      const parity = o.kind === "parity";
      const decodedEqual = parity && decodedSha ? (await rgbSha256(frame)) === decodedSha : undefined;
      const check = parity && n < o.pixelChecks;
      const reference = check ? new Uint8Array(await (await apiFetch(`/api/bench/${o.set}/${p.index}/input`)).arrayBuffer()) : null;
      const [w, h] = [frame.width, frame.height];
      const r = await detector.detect(frame, conf, check);
      records.push({
        index: p.index, size: [w, h], input: [1, 3, r.inputHeight, r.inputWidth], ms: r.ms,
        boxes: r.boxes.map((b) => [b.cls, r5(b.x1 / w), r5(b.y1 / h), r5(b.x2 / w), r5(b.y2 / h), r5(b.conf)]),
        ...(decodedEqual !== undefined ? { decoded_equal: decodedEqual } : {}),
        ...(reference && r.input ? { pixel_check: compareInput(r.input, reference) } : {}),
      });
      progress(n + 1, photos.length);
    }
  } finally {
    detector.close();
  }

  const medianMs = Object.fromEntries((["letterbox", "tensor", "run", "post", "total"] as const)
    .map((k) => [k, median(records.map((p) => p.ms[k]))])) as unknown as StageMs;
  const decoded = records.filter((p) => p.decoded_equal !== undefined);
  const body = {
    set: o.set, variant: o.variant, backend: o.backend, bench: o.kind, conf, threshold: config.threshold,
    options: o, rest_seconds: restSeconds, input: [1, 3, SQUARE, SQUARE], tight: false, started, ended: new Date().toISOString(),
    device, detector: detector.info, runtime: { "onnxruntime-web": detector.info.ortVersion },
    model: { variant: o.variant, sha256: model.sha256, bytes: model.bytes }, app: APP,
    median_ms: medianMs, photos: records,
  };
  const { file } = await postRecord(o.kind === "parity" ? "bench-parity" : "bench-speed", body);
  return {
    file, photos: records.length, medianMs, info: detector.info,
    boxesAtThreshold: records.reduce((n, p) => n + p.boxes.filter((b) => b[5] >= config.threshold).length, 0),
    decodedEqual: decoded.length ? decoded.filter((p) => p.decoded_equal).length : null,
    pixelChecks: records.flatMap((p) => (p.pixel_check ? [p.pixel_check] : [])),
  };
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length === 0 ? NaN : s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** SHA-256 of a bitmap's RGB pixels, as the API hashes PIL's decode. Drawn
 *  at its own size, so nothing is resampled. */
async function rgbSha256(bitmap: ImageBitmap): Promise<string> {
  const { width, height } = bitmap;
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("no 2D context");
  ctx.drawImage(bitmap, 0, 0);
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const rgb = new Uint8Array(width * height * 3);
  for (let p = 0, i = 0, j = 0; p < width * height; p++, i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", rgb));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The model's input as the browser made it (RGBA) against Ultralytics' (RGB). */
export function compareInput(rgba: Uint8ClampedArray, rgb: Uint8Array): PixelCheck {
  if (rgba.length / 4 !== rgb.length / 3) throw new Error(`${rgba.length / 4} pixels against ${rgb.length / 3}`);
  let max = 0;
  let sum = 0;
  let differing = 0;
  let over2 = 0;
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(rgba[i + c] - rgb[j + c]);
      sum += d;
      if (d > max) max = d;
      if (d) differing++;
      if (d > 2) over2++;
    }
  }
  const n = rgb.length;
  return { max, mean: sum / n, differing: differing / n, over2: over2 / n };
}
