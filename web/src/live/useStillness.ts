/**
 * The stillness gate's eye on the video (docs/m2c-live-boxes.md, "The frame
 * loop"): for each new video frame (requestVideoFrameCallback), the frame is
 * drawn into a 64 px copy and its motion value taken against the last copy
 * (stillness.ts). The gate's own cost a frame is timed, from drawing to the
 * value. It runs on every `every`-th frame (S3's check: every frame if it
 * costs under 2 ms on the phone, else every other).
 *
 * The copy is drawn on a canvas on the GPU, and only the 64 px result is
 * read back: S3's gate bench on the OnePlus 13 timed this at 11.5 ms a
 * frame, against 64 ms for a canvas kept on the CPU (willReadFrequently),
 * which first brings the whole camera frame to the CPU (gateBench.ts). The
 * frame is averaged down (imageSmoothingQuality "medium", mipmaps), not
 * sampled at a few points, so a hand's tremor over fine print doesn't look
 * like motion. The copy never reaches the model.
 */
import type { RefObject } from "react";
import { useEffect, useRef } from "react";
import { copyContext } from "./copiers";
import type { CopyName } from "./gateBench";
import { GATE_SIDE, centredGrey, gateSize, motionValue } from "./stillness";

export const GATE_COPY_NAME: CopyName = "gpu-medium";
export const GATE_COPY = { side: GATE_SIDE, copy: GATE_COPY_NAME, quality: "medium", canvas: "OffscreenCanvas on the GPU" };

/** The gate's 64 px copies of the video, the newest first. */
export class GateCopy {
  private ctx: OffscreenCanvasRenderingContext2D | null = null;
  private size: [number, number] = [0, 0];
  private history: Float32Array[] = [];
  private readonly depth: number;

  /** `depth`: how many earlier copies to compare with (the Motion page keeps 2). */
  constructor(depth = 1) {
    this.depth = depth;
  }

  /** The video's frame drawn and compared: values[k] against the copy k + 1 reads back
   *  (null where there is none yet); `cost` times the draw, the read and values[0]. */
  read(video: HTMLVideoElement): { values: (number | null)[]; cost: number } | null {
    const t0 = performance.now();
    const [w, h] = gateSize(video.videoWidth, video.videoHeight);
    if (!w) return null;
    if (w !== this.size[0] || h !== this.size[1] || !this.ctx) {
      this.ctx = copyContext(GATE_COPY_NAME, w, h);
      this.size = [w, h];
      this.history = []; // the frame turned: the old copies are another shape
    }
    this.ctx.drawImage(video, 0, 0, w, h);
    const grey = centredGrey(this.ctx.getImageData(0, 0, w, h).data);
    const first = this.history[0] ? motionValue(grey, this.history[0]) : null;
    const cost = performance.now() - t0;
    const values = [first];
    for (let k = 1; k < this.depth; k++) values.push(this.history[k] ? motionValue(grey, this.history[k]) : null);
    this.history = [grey, ...this.history].slice(0, this.depth);
    return { values, cost };
  }

  reset(): void {
    this.history = [];
  }
}

type FrameMeta = { presentedFrames: number };
type VideoFrames = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: (now: number, meta: FrameMeta) => void) => number;
  cancelVideoFrameCallback?: (handle: number) => void;
};

/**
 * Calls `onFrame` for each new video frame while `active`, with the time
 * (performance.now()) and how many frames the video presented since the
 * last call (1: none missed). Without requestVideoFrameCallback, every
 * animation frame instead (a repeated frame then looks still).
 */
export function useFrames(video: RefObject<HTMLVideoElement | null>, active: boolean,
  onFrame: (now: number, presented: number) => void): void {
  const handler = useRef(onFrame);
  useEffect(() => {
    handler.current = onFrame;
  });
  useEffect(() => {
    const v = video.current as VideoFrames | null;
    if (!v || !active) return;
    let stopped = false;
    let handle = 0;
    if (v.requestVideoFrameCallback) {
      let last = -1;
      const tick = (_: number, meta: FrameMeta) => {
        if (stopped) return;
        const presented = last < 0 ? 1 : meta.presentedFrames - last;
        last = meta.presentedFrames;
        handler.current(performance.now(), presented);
        handle = v.requestVideoFrameCallback!(tick);
      };
      handle = v.requestVideoFrameCallback(tick);
      return () => {
        stopped = true;
        v.cancelVideoFrameCallback?.(handle);
      };
    }
    const tick = () => {
      if (stopped) return;
      handler.current(performance.now(), 1);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => {
      stopped = true;
      cancelAnimationFrame(handle);
    };
  }, [video, active]);
}

export interface GateReading {
  now: number;
  skipped: boolean; // left out by the cadence: no value read
  value: number | null; // against the gate's last copy
  value2: number | null; // against the one before (kept when depth is 2)
  cost: number | null; // the gate's own ms for this frame
  presented: number;
}

/** The gate's readings of the video while `active`: every `every`-th frame read, the rest skipped. */
export function useStillness(video: RefObject<HTMLVideoElement | null>, active: boolean, every: number,
  onReading: (r: GateReading) => void, depth = 1): void {
  const copy = useRef<GateCopy | null>(null);
  const count = useRef(0);
  useEffect(() => {
    copy.current = new GateCopy(depth);
    count.current = 0;
  }, [active, every, depth]);
  useFrames(video, active, (now, presented) => {
    const v = video.current;
    const c = copy.current;
    if (!v || !c) return;
    if (count.current++ % every !== 0) {
      onReading({ now, skipped: true, value: null, value2: null, cost: null, presented });
      return;
    }
    const read = c.read(v);
    onReading({ now, skipped: false, value: read?.values[0] ?? null, value2: read?.values[1] ?? null,
      cost: read?.cost ?? null, presented });
  });
}
