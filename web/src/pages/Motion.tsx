/**
 * Motion (docs/m2c-live-boxes.md, "The app"; rule G): the gate's motion
 * values while the phone is held still on a pack for 20 s, then moved
 * between packs for 20 s. The laptop reads T from the record by rule G
 * (live.gate). Each frame keeps its value against the frame before (value1)
 * and the one before that (value2), so T can be read for a gate on every
 * frame or every other from the one record. Nothing here runs the model.
 */
import { useEffect, useRef, useState } from "react";
import { useCamera } from "../live/useCamera";
import { useVisible, useWakeLock } from "../live/usePage";
import type { GateReading } from "../live/useStillness";
import { GATE_COPY, useStillness } from "../live/useStillness";
import { percentile } from "../live/stats";
import { APP, deviceInfo, postError, postRecord } from "../records";
import { GateBench } from "./GateBench";

// Each action starts in the 3 s before its phase, which aren't kept: the first
// laptop record (S3) had 0.4 s of "moving" before the hand moved, a third of
// the moving frames under the still p95.
type Phase = "ready" | "still" | "between" | "moving";
const PHASES: [Phase, number, string][] = [
  ["ready", 3, "aim at a pack and hold still now"],
  ["still", 20, "keep still on the pack"],
  ["between", 3, "start moving between packs now"],
  ["moving", 20, "keep moving between packs, don't stop on one"],
];

interface Row {
  t: number;
  phase: "still" | "moving";
  value1: number | null;
  value2: number | null;
  cost: number | null;
  presented: number;
}

const r3 = (v: number | null) => (v === null ? null : Math.round(v * 1000) / 1000);

function spread(xs: number[]) {
  return { n: xs.length, p5: percentile(xs, 5), median: percentile(xs, 50), p95: percentile(xs, 95) };
}

export function Motion() {
  const [on, setOn] = useState(false);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const camera = useCamera(on, deviceId);
  const video = useRef<HTMLVideoElement>(null);
  const visible = useVisible();
  const wakeLock = useWakeLock(on);
  const wakeLockNow = useRef(wakeLock); // read when a record is posted, not when it was started
  useEffect(() => {
    wakeLockNow.current = wakeLock;
  }, [wakeLock]);
  const [phase, setPhase] = useState<{ name: Phase; left: number; hint: string } | null>(null);
  const [value, setValue] = useState<number | null>(null);
  const [result, setResult] = useState<{ file: string; summary: ReturnType<typeof summarise> } | null>(null);
  const [error, setError] = useState("");
  const rows = useRef<Row[]>([]);
  const current = useRef<Phase | null>(null);
  const shown = useRef(0);
  const hidden = useRef(false);
  const [benching, setBenching] = useState(false); // the gate bench runs: this page's own gate pauses

  useEffect(() => {
    const v = video.current;
    if (!v || !camera.stream) return;
    v.srcObject = camera.stream;
    v.play().catch((e) => setError(`video: ${String(e)}`));
  }, [camera.stream]);

  useEffect(() => {
    if (!visible && current.current) hidden.current = true;
  }, [visible]);

  useStillness(video, Boolean(camera.stream) && visible && !benching, 1, (r: GateReading) => {
    const p = current.current;
    if ((p === "still" || p === "moving") && !r.skipped) {
      rows.current.push({ t: r.now, phase: p, value1: r.value, value2: r.value2, cost: r.cost, presented: r.presented });
    }
    if (r.now - shown.current > 250) {
      shown.current = r.now;
      setValue(r.value);
    }
  }, 2);

  const record = async () => {
    setResult(null);
    setError("");
    rows.current = [];
    hidden.current = false;
    const started = new Date().toISOString();
    const t0 = performance.now();
    for (const [name, seconds, hint] of PHASES) {
      current.current = name;
      for (let left = seconds; left > 0; left--) {
        setPhase({ name, left, hint });
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
    current.current = null;
    setPhase(null);
    const data = rows.current;
    const summary = summarise(data);
    try {
      const { file } = await postRecord("motion", {
        started, ended: new Date().toISOString(), app: APP, device: await deviceInfo(),
        camera: { label: camera.label, settings: camera.settings, focus: camera.focus, cameras: camera.devices.length },
        gate: GATE_COPY, phases: Object.fromEntries(PHASES.map(([n, s]) => [n, s])), hidden: hidden.current,
        wake_lock: wakeLockNow.current, summary,
        frames: {
          t: data.map((d) => Math.round((d.t - t0) * 10) / 10), phase: data.map((d) => d.phase),
          value1: data.map((d) => r3(d.value1)), value2: data.map((d) => r3(d.value2)),
          cost: data.map((d) => r3(d.cost)), presented: data.map((d) => d.presented),
        },
      });
      setResult({ file, summary });
    } catch (e) {
      setError(String(e));
      postError("motion", e);
    }
  };

  const busy = phase !== null || benching;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <button className="rounded bg-cyan-600 px-3 py-1.5 font-medium text-white" disabled={busy} onClick={() => setOn(!on)}>
          {on ? "Stop camera" : "Start camera"}
        </button>
        {camera.stream && (
          <button className="rounded bg-emerald-600 px-3 py-1.5 font-medium text-white disabled:opacity-50" disabled={busy} onClick={record}>
            Record motion (46 s)
          </button>
        )}
        {camera.devices.length > 1 && (
          <select className="max-w-48 rounded bg-slate-800 px-2 py-1.5" value={deviceId ?? ""} disabled={busy}
            onChange={(e) => setDeviceId(e.target.value || null)}>
            <option value="">back camera</option>
            {camera.devices.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label}</option>)}
          </select>
        )}
      </div>

      <div className="relative w-full overflow-hidden rounded bg-black" style={{ height: "60vh" }}>
        <video ref={video} className="h-full w-full object-contain" playsInline muted />
        {phase && (
          <div className="absolute inset-x-0 top-0 bg-black/60 p-3 text-center text-lg font-semibold text-white">
            {phase.hint}: {phase.left} s
          </div>
        )}
      </div>

      <p className="font-mono text-xs text-slate-300">
        motion {value === null ? "–" : value.toFixed(2)} grey levels · {camera.settings ? `${camera.settings.width}×${camera.settings.height}` : ""}
        {camera.label ? ` · ${camera.label}` : ""} · wake lock {wakeLock}
      </p>
      {result && (
        <div className="rounded bg-slate-800 p-3 font-mono text-xs leading-relaxed">
          <div>posted: {result.file}</div>
          {(["still", "moving"] as const).map((p) => (["value1", "value2"] as const).map((k) => {
            const s = result.summary[p][k];
            return <div key={p + k}>{p} {k}: {s.n} frames · p5 {s.p5.toFixed(2)} · median {s.median.toFixed(2)} · p95 {s.p95.toFixed(2)}</div>;
          }))}
          <div>gate ms a frame: median {result.summary.cost.median.toFixed(2)} · p90 {result.summary.cost.p90.toFixed(2)} · max {result.summary.cost.max.toFixed(2)} · {result.summary.cost.n} frames</div>
        </div>
      )}
      {error && <p className="text-sm text-red-400">{error}</p>}
      {camera.error && <p className="text-sm text-red-400">{camera.error}</p>}
      <GateBench video={video} camera={camera} visible={visible} wakeLock={wakeLockNow} disabled={phase !== null}
        onRunning={setBenching} />
    </div>
  );
}

function summarise(rows: Row[]) {
  const values = (p: Row["phase"], k: "value1" | "value2") =>
    rows.flatMap((r) => (r.phase === p && r[k] !== null ? [r[k] as number] : []));
  const costs = rows.flatMap((r) => (r.cost === null ? [] : [r.cost]));
  return {
    still: { value1: spread(values("still", "value1")), value2: spread(values("still", "value2")) },
    moving: { value1: spread(values("moving", "value1")), value2: spread(values("moving", "value2")) },
    cost: { n: costs.length, median: percentile(costs, 50), p90: percentile(costs, 90), max: costs.length ? Math.max(...costs) : NaN },
  };
}
