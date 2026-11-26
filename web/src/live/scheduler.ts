/**
 * When a detection starts (docs/m2c-live-boxes.md, "The frame loop"), and
 * the loop as a whole: the gate, the scheduler and the tracker together.
 *
 * - One detection at a time, on the newest frame: frames that come while
 *   one runs are dropped, never queued.
 * - Only when the gate says still, and at least 200 ms after the last start.
 * - Rest: after 3 detections in a row while the picture stays still (no
 *   frame over T), one a second until a frame over T. The plan first asked
 *   for 3 with the same boxes; S3's soak found the boxes flicker on a still
 *   pack (the count changed between 76% of detections, 5 s of rest in 300),
 *   so the gate alone decides. Whether a detection found the same boxes as
 *   the one before (as many of each class, each paired at IoU 0.5 or more
 *   by the tracker's match) is still reported, for the records.
 * - A result whose detection started before the boxes were last cleared is
 *   stale: its picture is gone, so it is dropped.
 *
 * No React and no browser: useDetections.ts drives it from the camera.
 */
import type { GateState } from "./stillness";
import { StillnessGate } from "./stillness";
import type { Box, Track } from "./tracker";
import { Tracker, matchBoxes } from "./tracker";

export const MIN_GAP_MS = 200; // between detection starts: 5 a second at most
export const REST_GAP_MS = 1000; // while resting: one a second
export const REST_AFTER = 3; // detections in a row while the picture stays still
export const ALIKE_IOU = 0.5;

/** What the loop does with a frame: start a detection on it, or why not. */
export type Decision = "start" | "gated" | "busy" | "spaced" | "resting";

/** Whether two detections found the same boxes: as many of each class, each paired at IoU >= ALIKE_IOU. */
export function alike(a: readonly Box[], b: readonly Box[], minIou = ALIKE_IOU): boolean {
  return a.length === b.length && matchBoxes(a, b, minIou).length === a.length;
}

export interface Ticket {
  readonly id: number;
  readonly at: number; // when it started
  readonly clears: number; // the gate's clear count then
  readonly rested: boolean; // the scheduler was resting then
}

export interface Finished {
  stale: boolean; // started before the boxes were cleared: dropped
  alike: boolean; // the same boxes as the detection before
  resting: boolean; // resting from now on
}

export class Scheduler {
  private readonly rest: boolean;
  private running: Ticket | null = null;
  private lastStart = -Infinity;
  private previous: readonly Box[] | null = null;
  private inARow = 0; // detections finished since the picture last went over T
  private nextId = 1;

  /** `rest` false: never rest (the gate is off, so nothing could wake it). */
  constructor(rest = true) {
    this.rest = rest;
  }

  get busy(): boolean {
    return this.running !== null;
  }

  get streak(): number {
    return this.inARow;
  }

  get resting(): boolean {
    return this.rest && this.inARow >= REST_AFTER;
  }

  /** A frame (or a detection's end) with the gate's state: may a detection start now?
   *  A frame over T also ends any rest. */
  frame(gate: GateState, now: number): Decision {
    if (!gate.below) {
      this.inARow = 0;
      this.previous = null;
    }
    if (!gate.still) return "gated";
    if (this.running) return "busy";
    const gap = this.resting ? REST_GAP_MS : MIN_GAP_MS;
    if (now - this.lastStart < gap) return this.resting ? "resting" : "spaced";
    return "start";
  }

  start(now: number, clears: number): Ticket {
    if (this.running) throw new Error("a detection is running: one at a time");
    this.running = { id: this.nextId++, at: now, clears, rested: this.resting };
    this.lastStart = now;
    return this.running;
  }

  /** A detection's boxes, with the gate's clear count now. */
  finish(ticket: Ticket, boxes: readonly Box[], clears: number): Finished {
    this.end(ticket);
    if (clears !== ticket.clears) return { stale: true, alike: false, resting: this.resting };
    const same = this.previous !== null && alike(this.previous, boxes);
    this.inARow++;
    this.previous = boxes;
    return { stale: false, alike: same, resting: this.resting };
  }

  fail(ticket: Ticket): void {
    this.end(ticket);
  }

  private end(ticket: Ticket): void {
    if (this.running?.id !== ticket.id) throw new Error(`detection ${ticket.id} is not running`);
    this.running = null;
  }
}

export interface FrameOutcome {
  state: GateState;
  decision: Decision;
  cleared: boolean; // boxes were shown and this frame cleared them
}

/** The gate, the scheduler and the tracker, fed the motion value of each frame and each detection's boxes. */
export class FrameLoop {
  readonly gate: StillnessGate;
  readonly scheduler: Scheduler;
  readonly tracker = new Tracker();
  private latest: GateState | null = null;

  /** `t` null: the gate is off (rule G), so detection runs continuously and never rests. */
  constructor(t: number | null) {
    this.gate = new StillnessGate(t);
    this.scheduler = new Scheduler(t !== null);
  }

  get boxes(): readonly Track[] {
    return this.tracker.boxes;
  }

  get busy(): boolean {
    return this.scheduler.busy;
  }

  get state(): GateState | null {
    return this.latest;
  }

  frame(value: number | null, now: number): FrameOutcome {
    const state = this.gate.update(value, now);
    this.latest = state;
    const cleared = state.clear ? this.tracker.clear() : false;
    return { state, decision: this.scheduler.frame(state, now), cleared };
  }

  /** At a detection's end: may the next start now, on the frame on screen (the latest gate state)? */
  next(now: number): Decision {
    return this.latest ? this.scheduler.frame(this.latest, now) : "gated";
  }

  start(now: number): Ticket {
    return this.scheduler.start(now, this.gate.clears);
  }

  /** A detection's boxes: shown through the tracker unless stale. */
  finish(ticket: Ticket, boxes: readonly Box[]): Finished {
    const out = this.scheduler.finish(ticket, boxes, this.gate.clears);
    if (!out.stale) this.tracker.update(boxes);
    return out;
  }

  fail(ticket: Ticket): void {
    this.scheduler.fail(ticket);
  }

  /** The page was hidden: the picture is unknown. Boxes cleared, the gate starts over, and a
   *  detection still running in the worker comes back stale (it stays running: never two at once). */
  reset(): void {
    this.gate.reset();
    this.tracker.clear();
    this.latest = null;
  }
}
