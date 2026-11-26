/**
 * The frame loop in the page (docs/m2c-live-boxes.md, "The frame loop"):
 * the gate reads each frame (useStillness), the loop (scheduler.ts) says
 * whether a detection starts on it, the detector worker runs it, and the
 * tracker's boxes come back for the overlay. One detection at a time, on
 * the newest frame; when one ends, the next may start at once on the frame
 * on screen then. The loop stops while the page is hidden and starts over
 * when it is back. A LiveRecorder, when given, keeps every frame and
 * detection for the live record.
 */
import type { RefObject } from "react";
import { useEffect, useRef, useState } from "react";
import type { GateConfig } from "../api";
import type { Detector } from "../detector/client";
import type { DetectResult } from "../detector/types";
import type { LiveRecorder } from "./liveRecord";
import { FrameLoop } from "./scheduler";
import type { Motion } from "./stillness";
import { median } from "./stats";
import type { Track } from "./tracker";
import { useVisible } from "./usePage";
import { useStillness } from "./useStillness";

export interface LiveView {
  boxes: readonly Track[];
  frame: [number, number]; // the frame the boxes are in, px
  motion: Motion | "off"; // off: no gate (none pinned, or rule G turned it off)
  resting: boolean;
  last: (DetectResult & { bitmapMs: number }) | null;
  rate: number; // detections a second over the last 3 s
  gateMs: number | null; // the gate's median cost a frame, over the last 60 frames it read
}

const EMPTY: LiveView = { boxes: [], frame: [0, 0], motion: "settling", resting: false, last: null, rate: 0, gateMs: null };

interface Options {
  detector: Detector | null;
  video: RefObject<HTMLVideoElement | null>;
  threshold: number;
  gate: GateConfig | null;
  active: boolean; // the camera is on
  recorder: RefObject<LiveRecorder | null>;
  onError: (where: string, error: unknown) => void;
}

export function useDetections({ detector, video, threshold, gate, active, recorder, onError }: Options): LiveView {
  const t = gate?.on ? gate.t : null;
  const every = gate?.every ?? 1;
  const visible = useVisible();
  const loop = useRef<FrameLoop | null>(null);
  const [view, setView] = useState<LiveView>(EMPTY);
  const ends = useRef<number[]>([]);
  const costs = useRef<number[]>([]);
  const failed = useRef(false);

  useEffect(() => {
    loop.current = detector ? new FrameLoop(t) : null;
    failed.current = false;
    setView({ ...EMPTY, motion: t === null ? "off" : "settling" });
  }, [detector, t]);

  // Hidden: the loop stops (useStillness isn't active). Back: the picture is unknown, so it starts over.
  useEffect(() => {
    recorder.current?.event(performance.now(), visible ? "visible" : "hidden");
    if (!visible) {
      loop.current?.reset();
      setView((v) => ({ ...v, boxes: [] }));
    }
  }, [visible, recorder]);

  const run = async (l: FrameLoop) => {
    const v = video.current;
    if (!v || !detector || detector.busy || failed.current) return;
    const start = performance.now();
    const ticket = l.start(start);
    const row = { start, bitmap: 0, boxes: 0, dates: 0, stale: false, alike: false, rested: ticket.rested };
    try {
      const frame = await createImageBitmap(v);
      row.bitmap = performance.now() - start;
      const r = await detector.detect(frame, threshold);
      const end = performance.now();
      const out = l.finish(ticket, r.boxes);
      recorder.current?.detection({ ...row, end, ms: r.ms, boxes: r.boxes.length, dates: r.boxes.filter((b) => b.cls === 0).length,
        stale: out.stale, failed: false, alike: out.alike });
      ends.current = [...ends.current.filter((e) => end - e <= 3000), end];
      const n = ends.current.length;
      const rate = n > 1 ? ((n - 1) * 1000) / (end - ends.current[0]) : 0;
      if (loop.current !== l) return; // the detector changed meanwhile
      setView((view) => ({ ...view, boxes: l.boxes, frame: [r.frameWidth, r.frameHeight], resting: out.resting,
        last: { ...r, bitmapMs: row.bitmap }, rate, gateMs: costs.current.length ? median(costs.current) : null }));
      if (l.next(performance.now()) === "start") void run(l);
    } catch (e) {
      l.fail(ticket);
      recorder.current?.detection({ ...row, end: performance.now(), ms: null, failed: true });
      if (!detector.closed) {
        failed.current = true; // no more starts on this detector: one report, not one a frame
        onError("live: detection", e);
      }
    }
  };

  useStillness(video, active && visible && detector !== null, every, (r) => {
    const l = loop.current;
    if (!l) return;
    if (r.skipped) {
      recorder.current?.skip(r.presented);
      return;
    }
    const out = l.frame(r.value, r.now);
    recorder.current?.frame({ t: r.now, value: r.value, cost: r.cost, decision: out.decision, motion: out.state.motion,
      resting: l.scheduler.resting, presented: r.presented });
    if (r.cost !== null) costs.current = [...costs.current.slice(-59), r.cost];
    if (out.decision === "start") void run(l);
    const motion = t === null ? "off" : out.state.motion;
    setView((view) => (view.motion === motion && view.resting === l.scheduler.resting && !out.cleared ? view
      : { ...view, motion, resting: l.scheduler.resting, boxes: l.boxes }));
  });

  return view;
}
