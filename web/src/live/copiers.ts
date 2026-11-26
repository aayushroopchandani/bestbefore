/**
 * The ways of making the gate's small copy of a video frame (gateBench.ts
 * names them). Each copier owns its canvas, made once at the copy's size,
 * and returns the copy's RGBA with `block`: the ms it held the main thread.
 * For the bitmap copier that leaves out the wait for createImageBitmap,
 * whose shrinking runs elsewhere; the time until the copy is ready is the
 * caller's to measure.
 */
import type { CopyName } from "./gateBench";

export interface Copied {
  rgba: Uint8ClampedArray;
  block: number;
}

export type Copier = (video: HTMLVideoElement) => Promise<Copied>;

/** The canvas a copy draws on: on the GPU for gpu-medium (only the small
 *  result is read back), else kept on the CPU (willReadFrequently). */
export function copyContext(copy: CopyName, width: number, height: number): OffscreenCanvasRenderingContext2D {
  const ctx = new OffscreenCanvas(width, height).getContext("2d", { willReadFrequently: copy !== "gpu-medium", alpha: false });
  if (!ctx) throw new Error(`no 2D context for the ${copy} copy`);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = copy === "cpu-low" ? "low" : "medium";
  return ctx;
}

export function makeCopier(copy: CopyName, width: number, height: number): Copier {
  const ctx = copyContext(copy, width, height);
  if (copy === "bitmap-medium") {
    return async (video) => {
      const t0 = performance.now();
      const pending = createImageBitmap(video, { resizeWidth: width, resizeHeight: height, resizeQuality: "medium" });
      const before = performance.now() - t0;
      const bitmap = await pending;
      const t1 = performance.now();
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      const rgba = ctx.getImageData(0, 0, width, height).data;
      return { rgba, block: before + performance.now() - t1 };
    };
  }
  return async (video) => {
    const t0 = performance.now();
    ctx.drawImage(video, 0, 0, width, height);
    const rgba = ctx.getImageData(0, 0, width, height).data;
    return { rgba, block: performance.now() - t0 };
  };
}
