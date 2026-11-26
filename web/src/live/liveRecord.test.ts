// What a live record holds and how its summary is read (docs/m2c-live-boxes.md,
// "Records and scores": frames seen, gated, dropped; detections a second while
// still; still-to-boxes times; a median a minute). Written before liveRecord.ts.
import { describe, expect, it } from "vitest";
import type { DetectionRow, FrameRow } from "./liveRecord";
import { LiveRecorder, summarise } from "./liveRecord";

const STAGES = { letterbox: 20, tensor: 12, run: 160, post: 2, total: 194 };

function frame(t: number, over: Partial<FrameRow> = {}): FrameRow {
  return { t, value: 1, cost: 1, decision: "busy", motion: "still", resting: false, presented: 1, ...over };
}

function detection(start: number, end: number, over: Partial<DetectionRow> = {}): DetectionRow {
  return { start, end, bitmap: 2, ms: STAGES, boxes: 1, dates: 1, stale: false, failed: false, alike: false,
    rested: false, ...over };
}

describe("summarise: frames", () => {
  it("counts every frame by what the loop did with it, the frames the gate's cadence left out too", () => {
    const frames = [
      frame(0, { decision: "gated", motion: "moving" }), frame(33, { decision: "gated", motion: "settling" }),
      frame(66, { decision: "start" }), frame(99, { decision: "busy" }), frame(132, { decision: "busy" }),
      frame(165, { decision: "spaced" }), frame(198, { decision: "resting" }),
    ];
    const s = summarise({ frames, skipped: 3, detections: [], startedAt: 0, endedAt: 231 });
    expect(s.frames).toEqual({ seen: 10, gated: 2, busy: 2, spaced: 1, resting: 1, started: 1, skipped: 3, gated_share: 0.2 });
  });

  it("gives the gate's own cost a frame: median, 90th percentile and maximum", () => {
    const frames = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((c, i) => frame(33 * i, { cost: c }));
    frames.push(frame(330, { cost: null, value: null })); // the first frame of a copy size: no value, no cost
    const s = summarise({ frames, skipped: 0, detections: [], startedAt: 0, endedAt: 363 });
    expect(s.gate_ms).toEqual({ frames: 10, median: 5.5, p90: 9.1, max: 10 });
  });

  it("counts the frames the page's callback never saw (presented frames past one)", () => {
    const frames = [frame(0, { presented: 1 }), frame(66, { presented: 2 }), frame(166, { presented: 3 })];
    expect(summarise({ frames, skipped: 0, detections: [], startedAt: 0, endedAt: 200 }).missed_frames).toBe(3);
  });

  it("counts the frames missed before a frame the cadence skipped too (S3's first scan lost them)", () => {
    const frames = [frame(0, { presented: 1 }), frame(133, { presented: 2 })];
    expect(summarise({ frames, skipped: 2, skippedMissed: 4, detections: [], startedAt: 0, endedAt: 200 }).missed_frames).toBe(5);
  });
});

describe("summarise: detections a second", () => {
  // Frames every 100 ms for one second, all still; the last five resting.
  const frames = Array.from({ length: 11 }, (_, i) => frame(100 * i, { resting: i >= 5 }));
  const detections = [
    detection(0, 200), detection(200, 400), detection(400, 600), detection(600, 800, { rested: true }),
  ];
  const s = summarise({ frames, skipped: 0, detections, startedAt: 0, endedAt: 1100 });

  it("adds the time between frames to still, and to awake or resting", () => {
    expect([s.detections.still_seconds, s.detections.awake_seconds, s.detections.resting_seconds]).toEqual([1, 0.5, 0.5]);
  });

  it("gives detections a second while still, and apart awake and resting", () => {
    expect(s.detections.per_second_still).toBe(4);
    expect(s.detections.per_second_awake).toBe(6);
    expect(s.detections.per_second_resting).toBe(2);
  });

  it("gives each stage's median over the detections that ran, the bitmap's too", () => {
    const failed = detection(900, 950, { failed: true, ms: null });
    const t = summarise({ frames, skipped: 0, detections: [...detections, failed], startedAt: 0, endedAt: 1100 });
    expect(t.median_ms).toEqual({ bitmap: 2, ...STAGES });
    expect([t.detections.n, t.detections.failed, t.detections.stale, t.detections.applied]).toEqual([5, 1, 0, 4]);
  });
});

describe("summarise: holding still to boxes", () => {
  // moving, then under T from 100 ms (settling), still from 300 ms, moving again at 900 ms
  const motions = ["moving", "settling", "settling", "still", "still", "still", "still", "still", "still", "moving"] as const;
  const frames = motions.map((m, i) => frame(100 * i, { motion: m }));

  it("times from the first frame under T to the end of the first detection started in that hold", () => {
    const detections = [detection(300, 520), detection(600, 800)];
    const s = summarise({ frames, skipped: 0, detections, startedAt: 0, endedAt: 1000 });
    expect(s.still_to_boxes_ms).toEqual([420]);
    expect([s.holds, s.holds_without_boxes]).toEqual([1, 0]);
  });

  it("skips a stale or failed detection, and one started before the hold", () => {
    const detections = [detection(-100, 350), detection(300, 500, { stale: true }), detection(500, 700, { failed: true, ms: null }),
      detection(700, 940)];
    const s = summarise({ frames, skipped: 0, detections, startedAt: 0, endedAt: 1000 });
    expect(s.still_to_boxes_ms).toEqual([840]);
  });

  it("counts a hold with no detection in it, and leaves out a hold already under way when the record began", () => {
    const more = [...frames, frame(1000, { motion: "settling" }), frame(1100, { motion: "still" }), frame(1200, { motion: "moving" })];
    const atStart = [frame(0, { motion: "still" }), ...frames.slice(1)]; // under T from the record's first frame
    expect(summarise({ frames: more, skipped: 0, detections: [], startedAt: 0, endedAt: 1300 }))
      .toMatchObject({ holds: 2, holds_without_boxes: 2, still_to_boxes_ms: [] });
    expect(summarise({ frames: atStart, skipped: 0, detections: [], startedAt: 0, endedAt: 1000 }).holds).toBe(0);
  });
});

describe("summarise: a median a minute", () => {
  it("groups detections by the minute they started in, from the record's start", () => {
    const detections = [
      detection(1_000, 1_200, { ms: { ...STAGES, total: 200, run: 150 } }),
      detection(30_000, 30_300, { ms: { ...STAGES, total: 300, run: 170 } }),
      detection(61_000, 61_250, { ms: { ...STAGES, total: 250, run: 180 } }),
    ];
    const s = summarise({ frames: [frame(500), frame(61_500)], skipped: 0, detections, startedAt: 500, endedAt: 61_500 });
    expect(s.by_minute).toEqual([
      { minute: 0, detections: 2, median_total: 250, median_run: 160 },
      { minute: 1, detections: 1, median_total: 250, median_run: 180 },
    ]);
  });
});

describe("LiveRecorder", () => {
  it("keeps frames from its start on, as columns of codes, and rounds what it keeps", () => {
    const r = new LiveRecorder(1000);
    r.frame(frame(999)); // before the record began: left out
    r.frame(frame(1000.04, { value: 1.23456, cost: 0.98765, decision: "gated", motion: "moving" }));
    r.frame(frame(1033.37, { value: null, cost: null, decision: "start", motion: "still", resting: true, presented: 2 }));
    r.skip(3); // two frames missed before this skipped one
    r.skip(1);
    expect(r.summary(1100).missed_frames).toBe(3);
    expect(r.columns()).toEqual({
      t: [0, 33.4], value: [1.235, null], cost: [0.988, null], decision: [1, 0], motion: [2, 0], resting: [0, 1],
      presented: [1, 2],
      codes: { decision: ["start", "gated", "busy", "spaced", "resting"], motion: ["still", "settling", "moving"] },
    });
    expect(r.skipped).toBe(2);
  });

  it("keeps detections that started from its start on, their times from the start", () => {
    const r = new LiveRecorder(1000);
    r.detection(detection(990, 1200));
    r.detection(detection(1200.26, 1400.5));
    expect(r.rows().map((d) => [d.start, d.end])).toEqual([[200.3, 400.5]]);
  });

  it("summarises what it kept", () => {
    const r = new LiveRecorder(1000);
    [0, 100, 200].forEach((t) => r.frame(frame(1000 + t, { decision: "gated", motion: "moving" })));
    expect(r.summary(1300).frames.gated).toBe(3);
  });
});
