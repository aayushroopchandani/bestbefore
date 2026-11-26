// When detections start (docs/m2c-live-boxes.md, "The frame loop"): one at
// a time on the newest frame, frames dropped and never queued, 200 ms
// between starts, one a second after 3 detections while the picture stays
// still, awake again on motion. Then the loop as a whole: the gate, the
// scheduler and the tracker together. Written before scheduler.ts
// (Decision 14). The rest rule first asked for 3 detections with the same
// boxes; S3's soak found the boxes flicker on a still pack (the count changed
// between 76% of detections), so the rest follows the gate alone.
import { describe, expect, it } from "vitest";
import type { GateState } from "./stillness";
import type { Box } from "./tracker";
import type { Ticket } from "./scheduler";
import { ALIKE_IOU, FrameLoop, MIN_GAP_MS, REST_AFTER, REST_GAP_MS, Scheduler, alike } from "./scheduler";

const box = (cls: number, x1: number, y1: number, x2: number, y2: number): Box => ({ cls, conf: 0.5, x1, y1, x2, y2 });
const DATE = [box(0, 100, 100, 200, 130)];

const still: GateState = { value: 1, below: true, still: true, motion: "still", dim: false, clear: false, clears: 0 };
const settling: GateState = { value: 1, below: true, still: false, motion: "settling", dim: true, clear: false, clears: 0 };
const moving: GateState = { value: 9, below: false, still: false, motion: "moving", dim: true, clear: false, clears: 0 };

/** Offers the scheduler a frame every 33 ms from `from` until before `to`; returns each decision. */
function frames(s: Scheduler, from: number, to: number, gate: GateState = still) {
  const out: string[] = [];
  for (let t = from; t < to; t += 33) out.push(s.frame(gate, t));
  return out;
}

/** One detection started at `at`, finished with `boxes` (no clearing between). */
function detect(s: Scheduler, at: number, boxes: Box[]) {
  return s.finish(s.start(at, 0), boxes, 0);
}

describe("Scheduler", () => {
  it("uses the plan's numbers: 200 ms between starts, one a second after 3 while still", () => {
    expect([MIN_GAP_MS, REST_GAP_MS, REST_AFTER, ALIKE_IOU]).toEqual([200, 1000, 3, 0.5]);
  });

  it("starts on a still frame, and holds frames that aren't still (gated)", () => {
    const s = new Scheduler();
    expect(s.frame(moving, 0)).toBe("gated");
    expect(s.frame(settling, 33)).toBe("gated");
    expect(s.frame(still, 66)).toBe("start");
  });

  it("never runs two at once: every frame during a detection is busy, and a second start is refused", () => {
    const s = new Scheduler();
    s.start(0, 0);
    expect(frames(s, 33, 600).every((d) => d === "busy")).toBe(true);
    expect(() => s.start(300, 0)).toThrow(/running/);
  });

  it("drops the frames that come during a detection and never queues them: the next starts on the first frame after", () => {
    const s = new Scheduler();
    const t = s.start(0, 0);
    const during = frames(s, 33, 600); // 18 frames while a 600 ms detection runs
    s.finish(t, DATE, 0);
    expect(during.filter((d) => d === "busy")).toHaveLength(18);
    expect(s.frame(still, 633)).toBe("start");
    s.start(633, 0);
    expect(s.frame(still, 666)).toBe("busy"); // one start, not a burst for the frames dropped
  });

  it("waits 200 ms between starts, however fast a detection ends", () => {
    const s = new Scheduler();
    detect(s, 0, DATE);
    expect(frames(s, 66, 200)).toEqual(["spaced", "spaced", "spaced", "spaced", "spaced"]);
    expect(s.frame(still, 199.9)).toBe("spaced");
    expect(s.frame(still, 200)).toBe("start");
  });

  it("rests after 3 detections in a row while the picture stays still, whatever their boxes: one a second until it moves", () => {
    const s = new Scheduler();
    expect(detect(s, 0, DATE).resting).toBe(false);
    expect(detect(s, 200, [...DATE, box(1, 10, 10, 50, 30)]).resting).toBe(false); // a box flickered in
    expect(detect(s, 400, [])).toEqual({ stale: false, alike: false, resting: true }); // and every box out
    expect(s.resting).toBe(true);
    expect(frames(s, 600, 1400).every((d) => d === "resting")).toBe(true);
    expect(s.frame(still, 1400)).toBe("start");
  });

  it("still says whether each detection found the same boxes as the one before, for the record", () => {
    const s = new Scheduler();
    expect(detect(s, 0, DATE).alike).toBe(false); // nothing before it
    expect(detect(s, 200, DATE).alike).toBe(true);
    expect(detect(s, 400, [...DATE, box(1, 10, 10, 50, 30)]).alike).toBe(false);
    expect(detect(s, 1400, [...DATE, box(1, 10, 10, 50, 30)]).alike).toBe(true);
    expect(s.streak).toBe(4);
  });

  it("starts counting again after a frame over T, even one that settles at once", () => {
    const s = new Scheduler();
    detect(s, 0, DATE);
    detect(s, 200, DATE);
    expect(s.frame(moving, 300)).toBe("gated");
    expect(s.streak).toBe(0);
    expect(s.frame(still, 450)).toBe("start");
    expect(detect(s, 450, DATE).resting).toBe(false);
    expect(s.streak).toBe(1);
  });

  it("wakes on motion: one frame over T ends the rest, and the next start is 200 ms after the last", () => {
    const s = new Scheduler();
    [0, 200, 400].forEach((t) => detect(s, t, DATE));
    expect(s.frame(still, 500)).toBe("resting");
    expect(s.frame(moving, 533)).toBe("gated");
    expect([s.resting, s.streak]).toEqual([false, 0]);
    expect(s.frame(settling, 566)).toBe("gated");
    expect(s.frame(still, 716)).toBe("start");
  });

  it("drops a result whose detection started before the boxes were cleared (stale), and doesn't count it", () => {
    const s = new Scheduler();
    detect(s, 0, DATE);
    const t = s.start(200, 0);
    expect(s.finish(t, DATE, 1)).toEqual({ stale: true, alike: false, resting: false }); // cleared once since
    expect(s.streak).toBe(1);
    expect(s.busy).toBe(false);
  });

  it("never rests when the gate is off (rule G): nothing could wake it", () => {
    const s = new Scheduler(false);
    [0, 200, 400, 600].forEach((t) => detect(s, t, DATE));
    expect(s.resting).toBe(false);
    expect(s.frame(still, 800)).toBe("start");
  });

  it("frees itself after a failed detection, its streak untouched, the gap counted from that start", () => {
    const s = new Scheduler();
    detect(s, 0, DATE);
    s.fail(s.start(200, 0));
    expect([s.busy, s.streak]).toEqual([false, 1]);
    expect(s.frame(still, 399)).toBe("spaced");
    expect(s.frame(still, 400)).toBe("start");
  });

  it("refuses to finish a ticket that isn't the running one", () => {
    const s = new Scheduler();
    const t = s.start(0, 0);
    s.finish(t, [], 0);
    expect(() => s.finish(t, [], 0)).toThrow(/not running/);
  });
});

describe("alike", () => {
  it("is true for as many boxes of each class, each paired at IoU 0.5 or more, in any order", () => {
    const a = [box(0, 0, 0, 100, 30), box(1, 200, 0, 260, 20)];
    const b = [box(1, 202, 1, 262, 21), box(0, 3, 0, 103, 31)];
    expect(alike(a, b)).toBe(true);
    expect(alike([], [])).toBe(true);
  });

  it("is false for a box more or fewer, a class changed, or a box moved past IoU 0.5", () => {
    const a = [box(0, 0, 0, 100, 30)];
    expect(alike(a, [])).toBe(false);
    expect(alike(a, [...a, box(0, 300, 0, 400, 30)])).toBe(false);
    expect(alike(a, [box(1, 0, 0, 100, 30)])).toBe(false);
    expect(alike(a, [box(0, 40, 0, 140, 30)])).toBe(false); // IoU 1,800 / 4,200
  });
});

describe("FrameLoop", () => {
  const T = 3;

  /** Frames every 33 ms with `value`; a detection of `ms` started whenever one may
   *  start, finished with `boxes` at the first frame after it ends (or at `to`). */
  function drive(loop: FrameLoop, from: number, to: number, value: number, ms: number, boxes: Box[]) {
    let running: { ticket: Ticket; ends: number } | null = null;
    const starts: number[] = [];
    for (let t = from; t < to; t += 33) {
      if (running && t >= running.ends) {
        loop.finish(running.ticket, boxes);
        running = null;
      }
      if (loop.frame(value, t).decision === "start") {
        running = { ticket: loop.start(t), ends: t + ms };
        starts.push(t);
      }
    }
    if (running) loop.finish(running.ticket, boxes);
    return starts;
  }

  it("shows a detection's boxes once the picture is still", () => {
    const loop = new FrameLoop(T);
    drive(loop, 0, 600, 1, 100, DATE);
    expect(loop.boxes.map((b) => [b.cls, b.x1])).toEqual([[0, 100]]);
  });

  it("dims the boxes while moving and clears them after 400 ms of motion, not at 300", () => {
    const loop = new FrameLoop(T);
    drive(loop, 0, 600, 1, 100, DATE);
    expect(loop.frame(9, 633).state.dim).toBe(true);
    for (let t = 666; t <= 933; t += 33) expect(loop.frame(9, t).cleared).toBe(false);
    expect(loop.boxes).toHaveLength(1); // 300 ms of motion: still shown, dimmed
    expect(loop.frame(9, 1033).cleared).toBe(true); // 400 ms
    expect(loop.boxes).toEqual([]);
  });

  it("drops a result that started before the boxes were cleared", () => {
    const loop = new FrameLoop(T);
    for (let t = 0; t < 300; t += 33) loop.frame(1, t);
    const ticket = loop.start(300);
    for (let t = 333; t < 1000; t += 33) loop.frame(9, t); // 667 ms of motion: cleared
    expect(loop.finish(ticket, DATE).stale).toBe(true);
    expect(loop.boxes).toEqual([]);
  });

  it("lets the next start at once when a detection ends, if still and 200 ms have passed", () => {
    const loop = new FrameLoop(T);
    for (let t = 0; t < 300; t += 33) loop.frame(1, t);
    loop.finish(loop.start(300), DATE);
    expect(loop.next(499)).toBe("spaced");
    expect(loop.next(528)).toBe("start");
  });

  it("runs continuously at the 200 ms gap when the gate is off", () => {
    const loop = new FrameLoop(null);
    const starts = drive(loop, 0, 2000, 50, 50, DATE);
    expect(starts.length).toBeGreaterThanOrEqual(9);
    starts.slice(1).forEach((t, i) => expect(t - starts[i]).toBeLessThan(MIN_GAP_MS + 33));
  });

  it("starts over after a reset (the page was hidden): no boxes, in-flight results stale, not still for 150 ms", () => {
    const loop = new FrameLoop(T);
    drive(loop, 0, 600, 1, 100, DATE);
    const ticket = loop.start(700);
    loop.reset();
    expect(loop.boxes).toEqual([]);
    expect(loop.busy).toBe(true); // the worker still runs it: never two at once
    expect(loop.finish(ticket, DATE).stale).toBe(true);
    expect(loop.frame(1, 5000).decision).toBe("gated");
  });
});
