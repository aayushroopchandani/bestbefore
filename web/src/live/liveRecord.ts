/**
 * What a live record keeps (docs/m2c-live-boxes.md, "Records and scores"):
 * every gate frame (its time, motion value, the gate's own cost, what the
 * loop did with it), every detection (its times by stage, how many boxes,
 * whether it was stale), and a summary read from them: frames seen, gated
 * and dropped; detections a second while still; the time from holding still
 * to boxes; the median a minute (heat). The raw columns go into the record
 * too, so Python can read it all again (S5).
 *
 * No React and no browser.
 */
import type { StageMs } from "../detector/types";
import type { Decision } from "./scheduler";
import type { Motion } from "./stillness";
import { median, percentile } from "./stats";

const DECISIONS: readonly Decision[] = ["start", "gated", "busy", "spaced", "resting"];
const MOTIONS: readonly Motion[] = ["still", "settling", "moving"];

/** One frame the gate looked at. Times in ms (performance.now()). */
export interface FrameRow {
  t: number;
  value: number | null; // the motion value (null: no copy to compare with)
  cost: number | null; // the gate's own ms for this frame: the copy drawn and read, grey, the value
  decision: Decision;
  motion: Motion;
  resting: boolean; // the scheduler was resting
  presented: number; // frames the video presented since the last callback (1: none missed)
}

/** One detection. */
export interface DetectionRow {
  start: number;
  end: number;
  bitmap: number; // ms to take the frame from the video (createImageBitmap)
  ms: StageMs | null; // the worker's stages; null if it failed
  boxes: number;
  dates: number;
  stale: boolean;
  failed: boolean;
  alike: boolean;
  rested: boolean; // it started while resting
}

export interface LiveSummary {
  seconds: number;
  frames: { seen: number; gated: number; busy: number; spaced: number; resting: number; started: number;
    skipped: number; gated_share: number };
  gate_ms: { frames: number; median: number; p90: number; max: number };
  missed_frames: number;
  detections: { n: number; applied: number; stale: number; failed: number; still_seconds: number;
    awake_seconds: number; resting_seconds: number; per_second_still: number; per_second_awake: number;
    per_second_resting: number };
  median_ms: Record<"bitmap" | keyof StageMs, number>;
  still_to_boxes_ms: number[];
  still_to_boxes_median: number;
  holds: number;
  holds_without_boxes: number;
  by_minute: { minute: number; detections: number; median_total: number; median_run: number }[];
}

export interface LiveData {
  frames: readonly FrameRow[];
  skipped: number; // frames the gate's cadence left out (it runs on every Nth frame)
  skippedMissed?: number; // frames the page never saw just before one of those
  detections: readonly DetectionRow[];
  startedAt: number;
  endedAt: number;
}

const round = (v: number, d: number) => Math.round(v * 10 ** d) / 10 ** d;
const per = (n: number, seconds: number) => (seconds > 0 ? round(n / seconds, 3) : 0);

export function summarise({ frames, skipped, skippedMissed = 0, detections, startedAt, endedAt }: LiveData): LiveSummary {
  const count = (d: Decision) => frames.filter((f) => f.decision === d).length;
  const seen = frames.length + skipped;
  const costs = frames.flatMap((f) => (f.cost === null ? [] : [f.cost]));

  // Time still, awake or resting: each frame's state lasts until the next frame.
  let still = 0;
  let resting = 0;
  for (let i = 0; i + 1 < frames.length; i++) {
    if (frames[i].motion !== "still") continue;
    const dt = frames[i + 1].t - frames[i].t;
    still += dt;
    if (frames[i].resting) resting += dt;
  }
  const ran = detections.filter((d) => !d.failed && d.ms);
  const stale = detections.filter((d) => d.stale).length;
  const failed = detections.filter((d) => d.failed).length;

  const stage = (k: keyof StageMs) => median(ran.map((d) => d.ms![k]));
  const median_ms = { bitmap: median(ran.map((d) => d.bitmap)), letterbox: stage("letterbox"),
    tensor: stage("tensor"), run: stage("run"), post: stage("post"), total: stage("total") };

  const holds = stillToBoxes(frames, detections);
  const minutes = new Map<number, DetectionRow[]>();
  for (const d of ran) {
    const m = Math.floor((d.start - startedAt) / 60_000);
    minutes.set(m, [...(minutes.get(m) ?? []), d]);
  }

  return {
    seconds: round((endedAt - startedAt) / 1000, 3),
    frames: { seen, gated: count("gated"), busy: count("busy"), spaced: count("spaced"), resting: count("resting"),
      started: count("start"), skipped, gated_share: seen ? round(count("gated") / seen, 4) : 0 },
    gate_ms: { frames: costs.length, median: median(costs), p90: percentile(costs, 90), max: costs.length ? Math.max(...costs) : NaN },
    missed_frames: frames.reduce((n, f) => n + Math.max(0, f.presented - 1), skippedMissed),
    detections: {
      n: detections.length, applied: detections.length - stale - failed, stale, failed,
      still_seconds: round(still / 1000, 3), awake_seconds: round((still - resting) / 1000, 3),
      resting_seconds: round(resting / 1000, 3),
      per_second_still: per(detections.length, still / 1000),
      per_second_awake: per(detections.filter((d) => !d.rested).length, (still - resting) / 1000),
      per_second_resting: per(detections.filter((d) => d.rested).length, resting / 1000),
    },
    median_ms,
    still_to_boxes_ms: holds.times,
    still_to_boxes_median: median(holds.times),
    holds: holds.count,
    holds_without_boxes: holds.count - holds.times.length,
    by_minute: [...minutes.entries()].sort((a, b) => a[0] - b[0]).map(([minute, ds]) => ({
      minute, detections: ds.length, median_total: median(ds.map((d) => d.ms!.total)),
      median_run: median(ds.map((d) => d.ms!.run)) })),
  };
}

/**
 * Holding still to boxes: for each hold (a run of still frames), from the
 * first frame under T before it (when the hand stopped) to the end of the
 * first detection started in it that wasn't stale or failed. A hold whose
 * run under T began at the record's first frame is left out: when the hand
 * stopped isn't known.
 */
function stillToBoxes(frames: readonly FrameRow[], detections: readonly DetectionRow[]) {
  const times: number[] = [];
  let count = 0;
  let belowSince: number | null = null;
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    belowSince = f.motion === "moving" ? null : (belowSince ?? f.t);
    const begins = f.motion === "still" && (i === 0 || frames[i - 1].motion !== "still");
    if (!begins || belowSince === frames[0].t) continue;
    let j = i;
    while (j + 1 < frames.length && frames[j + 1].motion === "still") j++;
    const ends = j + 1 < frames.length ? frames[j + 1].t : Infinity;
    count++;
    const first = detections.find((d) => !d.stale && !d.failed && d.start >= f.t && d.start < ends);
    if (first) times.push(round(first.end - belowSince!, 1));
  }
  return { times, count };
}

/** Collects a live record from `startedAt` on. */
export class LiveRecorder {
  readonly startedAt: number;
  private readonly frameRows: FrameRow[] = [];
  private readonly detectionRows: DetectionRow[] = [];
  private readonly eventRows: [number, string][] = [];
  private skippedFrames = 0;
  private skippedMissed = 0;

  constructor(startedAt: number) {
    this.startedAt = startedAt;
  }

  get skipped(): number {
    return this.skippedFrames;
  }

  frame(row: FrameRow): void {
    if (row.t >= this.startedAt) this.frameRows.push(row);
  }

  /** A frame the gate's cadence left out; `presented` as for a frame row (1: none missed before it). */
  skip(presented = 1): void {
    this.skippedFrames++;
    this.skippedMissed += Math.max(0, presented - 1);
  }

  detection(row: DetectionRow): void {
    if (row.start >= this.startedAt) this.detectionRows.push(row);
  }

  event(t: number, what: string): void {
    this.eventRows.push([round(t - this.startedAt, 1), what]);
  }

  /** The frames as columns, times from the start; decisions and motions as codes. */
  columns() {
    const f = this.frameRows;
    return {
      t: f.map((r) => round(r.t - this.startedAt, 1)),
      value: f.map((r) => (r.value === null ? null : round(r.value, 3))),
      cost: f.map((r) => (r.cost === null ? null : round(r.cost, 3))),
      decision: f.map((r) => DECISIONS.indexOf(r.decision)),
      motion: f.map((r) => MOTIONS.indexOf(r.motion)),
      resting: f.map((r) => (r.resting ? 1 : 0)),
      presented: f.map((r) => r.presented),
      codes: { decision: [...DECISIONS], motion: [...MOTIONS] },
    };
  }

  /** The detections, times from the start. */
  rows(): DetectionRow[] {
    const r1 = (v: number) => round(v, 1);
    return this.detectionRows.map((d) => ({
      ...d, start: r1(d.start - this.startedAt), end: r1(d.end - this.startedAt), bitmap: r1(d.bitmap),
      ms: d.ms && { letterbox: r1(d.ms.letterbox), tensor: r1(d.ms.tensor), run: r1(d.ms.run), post: r1(d.ms.post),
        total: r1(d.ms.total) },
    }));
  }

  events(): [number, string][] {
    return this.eventRows;
  }

  summary(endedAt: number): LiveSummary {
    return summarise({ frames: this.frameRows, skipped: this.skippedFrames, skippedMissed: this.skippedMissed,
      detections: this.detectionRows, startedAt: this.startedAt, endedAt });
  }
}
