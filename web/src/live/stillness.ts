/**
 * The stillness gate (docs/m2c-live-boxes.md, "The frame loop"): it decides
 * when the detector may run, never whether a date is readable (M4's job).
 *
 * Each frame the video is drawn into a 64 px copy (useStillness.ts). Here:
 * the copy's grey values minus their mean, so the camera's exposure moving
 * the brightness isn't motion; the motion value, the mean absolute
 * difference from the last copy; and the gate's timing: under T for 150 ms
 * is still, a detection may start; over T, the boxes dim; 400 ms of motion,
 * the boxes are cleared, since they no longer sit on the picture. T comes
 * from a Motion record by rule G (src/bestbefore/live.py).
 *
 * No React and no browser: the canvas work is useStillness.ts's.
 */

export const GATE_SIDE = 64; // the copy's long side, px
export const STILL_MS = 150; // under T this long: still
export const CLEAR_MS = 400; // over T this long: the boxes are cleared

/** The gate's copy size for a frame: 64 px on the long side, the short side in proportion (at least 1). */
export function gateSize(width: number, height: number, side = GATE_SIDE): [number, number] {
  if (!width || !height) return [0, 0];
  const scale = side / Math.max(width, height);
  return [Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale))];
}

/** RGBA bytes to grey (luma: 0.299 R + 0.587 G + 0.114 B), minus the mean. Alpha is ignored. */
export function centredGrey(rgba: ArrayLike<number>, out?: Float32Array): Float32Array {
  const n = rgba.length >> 2;
  const grey = out ?? new Float32Array(n);
  let sum = 0;
  for (let p = 0, i = 0; p < n; p++, i += 4) {
    const g = 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
    grey[p] = g;
    sum += g;
  }
  const mean = sum / n;
  for (let p = 0; p < n; p++) grey[p] -= mean;
  return grey;
}

/** The motion value: the mean absolute difference of two centred copies, in grey levels. */
export function motionValue(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`copies of ${a.length} and ${b.length} values: the frame changed size`);
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length;
}

export type Motion = "still" | "settling" | "moving";

export interface GateState {
  value: number | null; // this frame's motion value (null: no copy to compare with)
  below: boolean; // under T
  still: boolean; // under T for STILL_MS: a detection may start
  motion: Motion; // still; settling (under T, not yet for STILL_MS); moving (over T)
  dim: boolean; // the boxes drawn dim: whenever not still
  clear: boolean; // over T, CLEAR_MS since motion began: the boxes are cleared
  clears: number; // how many times the boxes have been cleared (or the gate reset): a detection
  //                 started before the latest is stale
}

export class StillnessGate {
  readonly t: number | null; // null: the gate is off (rule G found still and moving overlapping)
  private readonly stillMs: number;
  private readonly clearMs: number;
  private belowSince: number | null = null; // the first frame of the current run under T
  private movingSince: number | null = null; // the first frame over T since the gate was last still
  private clearedThisMotion = false;
  private clearCount = 0;

  constructor(t: number | null, stillMs = STILL_MS, clearMs = CLEAR_MS) {
    this.t = t;
    this.stillMs = stillMs;
    this.clearMs = clearMs;
  }

  get clears(): number {
    return this.clearCount;
  }

  update(value: number | null, now: number): GateState {
    if (this.t === null) {
      return { value, below: true, still: true, motion: "still", dim: false, clear: false, clears: this.clearCount };
    }
    const below = value !== null && value < this.t;
    if (below) this.belowSince ??= now;
    else this.belowSince = null;
    const still = this.belowSince !== null && now - this.belowSince >= this.stillMs;

    if (still) {
      this.movingSince = null;
      this.clearedThisMotion = false;
    } else if (!below) {
      this.movingSince ??= now;
    }
    const clear = !below && this.movingSince !== null && now - this.movingSince >= this.clearMs;
    if (clear && !this.clearedThisMotion) {
      this.clearedThisMotion = true;
      this.clearCount++;
    }
    const motion: Motion = still ? "still" : below ? "settling" : "moving";
    return { value, below, still, motion, dim: !still, clear, clears: this.clearCount };
  }

  /** Forgets the picture (the page was hidden): not still until STILL_MS under T again,
   *  and any detection started before is stale. */
  reset(): void {
    this.belowSince = null;
    this.movingSince = null;
    this.clearedThisMotion = false;
    this.clearCount++;
  }
}
