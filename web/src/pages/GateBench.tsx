/**
 * The gate bench (S3), on the Motion page: the ways of making the gate's
 * 64 px copy (gateBench.ts, copiers.ts), each for 3 s, two rounds, with the
 * phone propped on a pack. One read at a time, on the newest frame, as the
 * gate reads. Posts a gate-bench record: every read's cost and motion
 * value, the video's frames each copy saw or missed, and a summary per
 * copy. Nothing here runs the model; the Motion page's own gate pauses
 * meanwhile.
 */
import type { RefObject } from "react";
import { useEffect, useRef, useState } from "react";
import type { Copier } from "../live/copiers";
import { makeCopier } from "../live/copiers";
import type { BenchRow, CopyName, CopySummary, FrameCount, Slot } from "../live/gateBench";
import { BENCH_ROUNDS, BENCH_SECONDS, COPIES, benchPlan, slotAt, summariseBench } from "../live/gateBench";
import { centredGrey, gateSize, motionValue } from "../live/stillness";
import type { Camera } from "../live/useCamera";
import { useFrames } from "../live/useStillness";
import { APP, deviceInfo, postError, postRecord } from "../records";

const COUNTDOWN = 3;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const r3 = (v: number | null) => (v === null ? null : Math.round(v * 1000) / 1000);

interface Run {
  t0: number;
  plan: Slot[];
  copiers: Record<CopyName, Copier>;
  slot: Slot | null;
  prev: Float32Array | null; // the last copy in this slot
  inFlight: boolean;
  rows: BenchRow[];
  frames: Partial<Record<CopyName, FrameCount>>;
  failed: string;
}

interface Props {
  video: RefObject<HTMLVideoElement | null>;
  camera: Camera;
  visible: boolean;
  wakeLock: RefObject<string>;
  disabled: boolean;
  onRunning: (running: boolean) => void;
}

export function GateBench({ video, camera, visible, wakeLock, disabled, onRunning }: Props) {
  const [banner, setBanner] = useState<string | null>(null);
  const [active, setActive] = useState(false);
  const [result, setResult] = useState<{ file: string; summary: Partial<Record<CopyName, CopySummary>> } | null>(null);
  const [error, setError] = useState("");
  const run = useRef<Run | null>(null);
  const hidden = useRef(false);

  useEffect(() => {
    if (!visible && active) hidden.current = true;
  }, [visible, active]);

  useFrames(video, active && visible, (now, presented) => {
    const r = run.current;
    const v = video.current;
    if (!r || !v) return;
    const slot = slotAt(r.plan, now - r.t0);
    if (!slot) return;
    const f = (r.frames[slot.copy] ??= { callbacks: 0, presented: 0 });
    f.callbacks += 1;
    f.presented += presented;
    if (slot !== r.slot) {
      r.slot = slot;
      r.prev = null; // a new slot compares nothing with a copy seconds old
    }
    if (r.inFlight) return;
    r.inFlight = true;
    const start = performance.now();
    r.copiers[slot.copy](v)
      .then(({ rgba, block }) => {
        const g0 = performance.now();
        const grey = centredGrey(rgba);
        const same = r.slot === slot;
        const value = same && r.prev ? motionValue(grey, r.prev) : null;
        if (same) r.prev = grey;
        const end = performance.now();
        r.rows.push({ t: start - r.t0, copy: slot.copy, round: slot.round, block: block + (end - g0), wall: end - start, value });
      })
      .catch((e) => {
        r.failed ||= `${slot.copy}: ${String(e)}`;
      })
      .finally(() => {
        r.inFlight = false;
      });
  });

  const bench = async () => {
    const v = video.current;
    if (!v) return;
    setResult(null);
    setError("");
    hidden.current = false;
    onRunning(true);
    try {
      for (let left = COUNTDOWN; left > 0; left--) {
        setBanner(`prop the phone on a pack: starts in ${left} s`);
        await sleep(1000);
      }
      const [w, h] = gateSize(v.videoWidth, v.videoHeight);
      if (!w) throw new Error("the video has no size yet");
      const copiers = Object.fromEntries(COPIES.map((c) => [c, makeCopier(c, w, h)])) as Record<CopyName, Copier>;
      const plan = benchPlan();
      const end = plan.at(-1)!.to;
      const started = new Date().toISOString();
      const r: Run = { t0: performance.now(), plan, copiers, slot: null, prev: null, inFlight: false, rows: [], frames: {}, failed: "" };
      run.current = r;
      setActive(true);
      for (let ms = 0; ms < end; ms = performance.now() - r.t0) {
        const slot = slotAt(plan, ms);
        setBanner(`${slot?.copy}, round ${slot?.round} of ${BENCH_ROUNDS} · ${Math.ceil((end - ms) / 1000)} s left · keep it propped`);
        await sleep(250);
      }
      setActive(false);
      while (r.inFlight) await sleep(20);
      run.current = null;
      setBanner("posting the record…");
      const summary = summariseBench(r.rows, plan, r.frames);
      const rows = r.rows;
      const { file } = await postRecord("gate-bench", {
        started, ended: new Date().toISOString(), app: APP, device: await deviceInfo(),
        camera: { label: camera.label, settings: camera.settings, focus: camera.focus, cameras: camera.devices.length },
        size: [w, h], plan: { copies: COPIES, rounds: BENCH_ROUNDS, seconds: BENCH_SECONDS },
        hidden: hidden.current, wake_lock: wakeLock.current, failed: r.failed || null, summary, video_frames: r.frames,
        reads: {
          t: rows.map((x) => Math.round(x.t * 10) / 10), copy: rows.map((x) => x.copy), round: rows.map((x) => x.round),
          block: rows.map((x) => r3(x.block)), wall: rows.map((x) => r3(x.wall)), value: rows.map((x) => r3(x.value)),
        },
      });
      setResult({ file, summary });
    } catch (e) {
      setError(`gate bench: ${String(e)}`);
      postError("gate bench", e);
    } finally {
      run.current = null;
      setActive(false);
      setBanner(null);
      onRunning(false);
    }
  };

  const f2 = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : "–");
  return (
    <>
      <button className="rounded bg-violet-600 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
        disabled={disabled || banner !== null || !camera.stream} onClick={bench}>
        Bench the gate ({COUNTDOWN + COPIES.length * BENCH_ROUNDS * BENCH_SECONDS} s)
      </button>
      {banner && <div className="rounded bg-violet-900/70 p-2 text-center text-base font-semibold text-white">{banner}</div>}
      {result && (
        <div className="overflow-x-auto rounded bg-slate-800 p-3 font-mono text-xs leading-relaxed">
          <div>posted: {result.file}</div>
          <table className="mt-1">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="pr-3">copy</th><th className="pr-3">ms held</th><th className="pr-3">p90</th>
                <th className="pr-3">ms to ready</th><th className="pr-3">reads/s</th><th className="pr-3">missed</th>
                <th className="pr-3">value median</th><th>p95</th>
              </tr>
            </thead>
            <tbody>
              {COPIES.map((c) => {
                const s = result.summary[c];
                return s && (
                  <tr key={c}>
                    <td className="pr-3">{c}</td><td className="pr-3">{f2(s.block.median)}</td><td className="pr-3">{f2(s.block.p90)}</td>
                    <td className="pr-3">{f2(s.wall.median)}</td><td className="pr-3">{s.per_second.toFixed(1)}</td>
                    <td className="pr-3">{s.missed}/{s.frames}</td><td className="pr-3">{f2(s.value.median)}</td><td>{f2(s.value.p95)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
    </>
  );
}
