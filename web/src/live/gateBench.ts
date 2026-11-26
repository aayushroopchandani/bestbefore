/**
 * The gate bench (S3): ways of making the gate's 64 px copy, each timed on
 * the phone in turn, since S3's first phone Motion record found the copy
 * took 63 ms a frame. Each copy runs for a few seconds, round after round
 * (a second round shows whether one changes as it goes); the page records
 * every read and this sums them up. The copies themselves are in
 * copiers.ts: they need a browser.
 */
import { median, percentile } from "./stats";

/** cpu-*: a canvas the CPU reads back (willReadFrequently), the frame averaged down (medium) or
 *  sampled (low); gpu-medium: a canvas on the GPU, only the 64 px result read back;
 *  bitmap-medium: createImageBitmap shrinks the frame, then the small bitmap is read. */
export const COPIES = ["cpu-medium", "cpu-low", "gpu-medium", "bitmap-medium"] as const;
export type CopyName = (typeof COPIES)[number];
export const BENCH_ROUNDS = 2;
export const BENCH_SECONDS = 3;

export interface Slot {
  copy: CopyName;
  round: number; // from 1
  from: number; // ms from the bench's start, inclusive
  to: number; // exclusive
}

export function benchPlan(copies: readonly CopyName[] = COPIES, rounds = BENCH_ROUNDS, seconds = BENCH_SECONDS): Slot[] {
  const plan: Slot[] = [];
  let at = 0;
  for (let round = 1; round <= rounds; round++) {
    for (const copy of copies) {
      plan.push({ copy, round, from: at, to: at + seconds * 1000 });
      at += seconds * 1000;
    }
  }
  return plan;
}

export function slotAt(plan: readonly Slot[], ms: number): Slot | null {
  return plan.find((s) => s.from <= ms && ms < s.to) ?? null;
}

export interface BenchRow {
  t: number; // ms from the bench's start, when the read began
  copy: CopyName;
  round: number;
  block: number; // ms the main thread was held: the copy, the grey values and the motion value
  wall: number; // ms from the read's start until its motion value was ready
  value: number | null; // against the copy before, in the same slot (none for a slot's first read)
}

/** The video's frames in a copy's slots: `callbacks` reached the page, `presented` were shown. */
export interface FrameCount {
  callbacks: number;
  presented: number;
}

export interface CopySummary {
  reads: number;
  per_second: number;
  frames: number; // presented by the video in this copy's slots
  missed: number; // of those, never seen by the page (its main thread was busy)
  block: { median: number; p90: number; max: number };
  wall: { median: number };
  value: { n: number; median: number; p95: number };
}

export function summariseBench(rows: readonly BenchRow[], plan: readonly Slot[],
  frames: Partial<Record<CopyName, FrameCount>>): Partial<Record<CopyName, CopySummary>> {
  const out: Partial<Record<CopyName, CopySummary>> = {};
  for (const copy of COPIES) {
    const slots = plan.filter((s) => s.copy === copy);
    if (!slots.length) continue;
    const mine = rows.filter((r) => r.copy === copy);
    const seconds = slots.reduce((a, s) => a + (s.to - s.from), 0) / 1000;
    const block = mine.map((r) => r.block);
    const values = mine.flatMap((r) => (r.value === null ? [] : [r.value]));
    const f = frames[copy] ?? { callbacks: 0, presented: 0 };
    out[copy] = {
      reads: mine.length,
      per_second: mine.length / seconds,
      frames: f.presented,
      missed: f.presented - f.callbacks,
      block: { median: median(block), p90: percentile(block, 90), max: block.length ? Math.max(...block) : NaN },
      wall: { median: median(mine.map((r) => r.wall)) },
      value: { n: values.length, median: median(values), p95: percentile(values, 95) },
    };
  }
  return out;
}
