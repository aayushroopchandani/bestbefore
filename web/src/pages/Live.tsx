/**
 * Live (docs/m2c-live-boxes.md, "How the app reads the camera"): the camera,
 * the stillness gate, one detection at a time on the newest frame, the
 * tracker's boxes over the video, and a line of numbers. The gate's T comes
 * from the API (rule G on the pinned Motion record); with none, the gate is
 * off and detection runs continuously.
 *
 * Records (S3): "Record" keeps every frame and detection for a scan (60 s,
 * five packs about 5 s each) or a soak (5 minutes on one pack, the phone
 * propped up), then posts a live record. `rec=scan` or `rec=soak` in the
 * link picks one.
 */
import { useEffect, useRef, useState } from "react";
import type { ApiConfig } from "../api";
import { hashParams, keyHeaders } from "../api";
import { Detector, backendName } from "../detector/client";
import type { Backend } from "../detector/types";
import { LiveBoxes } from "../live/LiveBoxes";
import { LiveRecorder } from "../live/liveRecord";
import { ALIKE_IOU, MIN_GAP_MS, REST_AFTER, REST_GAP_MS } from "../live/scheduler";
import { CLEAR_MS, STILL_MS } from "../live/stillness";
import { MATCH_IOU, MAX_MISSES, SMOOTH } from "../live/tracker";
import { useCamera } from "../live/useCamera";
import { useDetections } from "../live/useDetections";
import { useWakeLock } from "../live/usePage";
import { GATE_COPY } from "../live/useStillness";
import { APP, deviceInfo, postError, postRecord } from "../records";

const SQUARE = 1280; // R1 kept square padding: every frame's input is 1,280 x 1,280
const COUNTDOWN = 3;
const KINDS = {
  scan: { seconds: 60, cue: (s: number) => `pack ${Math.min(5, Math.floor(s / 12) + 1)} of 5: aim, hold still about 5 s, then the next` },
  soak: { seconds: 300, cue: () => "propped up on one pack: leave it" },
} as const;
type Kind = keyof typeof KINDS;

export function Live({ config }: { config: ApiConfig }) {
  const hasGpu = "gpu" in navigator;
  const [backend, setBackend] = useState<Backend>(hashParams().get("backend") === "wasm" || !hasGpu ? "wasm" : "webgpu");
  const [variant, setVariant] = useState("fp32");
  const [on, setOn] = useState(false);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const camera = useCamera(on, deviceId);
  const video = useRef<HTMLVideoElement>(null);
  const [detector, setDetector] = useState<Detector | null>(null);
  const [status, setStatus] = useState("");
  const wakeLock = useWakeLock(on);
  const wakeLockNow = useRef(wakeLock); // read when a record is posted, not when it was started
  useEffect(() => {
    wakeLockNow.current = wakeLock;
  }, [wakeLock]);

  const [kind, setKind] = useState<Kind>(hashParams().get("rec") === "soak" ? "soak" : "scan");
  const recorder = useRef<LiveRecorder | null>(null);
  const [recording, setRecording] = useState<{ phase: "countdown" | "recording" | "posting"; left: number; elapsed: number } | null>(null);
  const stopAsked = useRef(false);
  const [posted, setPosted] = useState("");

  const view = useDetections({
    detector, video, threshold: config.threshold, gate: config.gate, active: Boolean(camera.stream), recorder,
    onError: (where, e) => {
      setStatus(`${where}: ${String(e)}`);
      postError(where, e, { backend, variant });
    },
  });

  useEffect(() => {
    const v = video.current;
    if (!v || !camera.stream) return;
    v.srcObject = camera.stream;
    v.play().catch((e) => setStatus(`video: ${String(e)}`));
  }, [camera.stream]);

  // The detector: started with the camera, again when the backend or precision changes.
  useEffect(() => {
    if (!on) return;
    let made: Detector | null = null;
    let cancelled = false;
    setStatus("loading the model…");
    Detector.start({
      modelUrl: config.models[variant].url, headers: keyHeaders(), backend, wasmPaths: "/ort/",
      inputWidth: SQUARE, inputHeight: SQUARE, warmups: 2,
    }).then((d) => {
      if (cancelled) d.close();
      else {
        made = d;
        setDetector(d);
        setStatus("");
      }
    }).catch((e) => {
      setStatus(`detector: ${String(e)}`);
      postError("live: detector", e, { backend, variant });
    });
    return () => {
      cancelled = true;
      made?.close();
      setDetector(null);
    };
  }, [on, backend, variant, config]);

  const record = async () => {
    if (!detector) return;
    const { seconds } = KINDS[kind];
    stopAsked.current = false;
    setPosted("");
    for (let left = COUNTDOWN; left > 0; left--) {
      setRecording({ phase: "countdown", left, elapsed: 0 });
      await new Promise((r) => setTimeout(r, 1000));
    }
    const started = new Date().toISOString();
    const t0 = performance.now();
    recorder.current = new LiveRecorder(t0);
    for (let s = 0; s < seconds && !stopAsked.current; s++) {
      setRecording({ phase: "recording", left: seconds - s, elapsed: s });
      await new Promise((r) => setTimeout(r, Math.max(0, t0 + 1000 * (s + 1) - performance.now())));
    }
    const ended = performance.now();
    const r = recorder.current;
    recorder.current = null;
    setRecording({ phase: "posting", left: 0, elapsed: seconds });
    try {
      const model = config.models[variant];
      const { file } = await postRecord("live", {
        label: kind, seconds: Math.round((ended - t0) / 100) / 10, seconds_asked: seconds, stopped_early: stopAsked.current,
        backend, variant, input: [1, 3, SQUARE, SQUARE], tight: false, threshold: config.threshold,
        gate: { config: config.gate, copy: GATE_COPY, every: config.gate?.every ?? 1 },
        loop: { still_ms: STILL_MS, clear_ms: CLEAR_MS, min_gap_ms: MIN_GAP_MS, rest_gap_ms: REST_GAP_MS,
          rest_after: REST_AFTER, alike_iou: ALIKE_IOU, match_iou: MATCH_IOU, smooth: SMOOTH, max_misses: MAX_MISSES },
        started, ended: new Date().toISOString(), app: APP, device: await deviceInfo(), detector: detector.info,
        runtime: { "onnxruntime-web": detector.info.ortVersion },
        model: { variant, sha256: model.sha256, bytes: model.bytes },
        camera: { label: camera.label, settings: camera.settings, focus: camera.focus, cameras: camera.devices.length },
        wake_lock: wakeLockNow.current, summary: r.summary(ended), events: r.events(), frames: r.columns(), detections: r.rows(),
      });
      setPosted(file);
    } catch (e) {
      setPosted(`not posted: ${String(e)}`);
      postError("live: record", e, { kind });
    }
    setRecording(null);
  };

  const s = camera.settings;
  const longSide = s?.width && s?.height ? Math.max(s.width, s.height) : 0;
  const info = detector?.info;
  const last = view.last;
  const busy = recording !== null;
  const motion = view.motion === "off" ? "gate off" : view.resting ? `${view.motion}, resting` : view.motion;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <button className="rounded bg-cyan-600 px-3 py-1.5 font-medium text-white disabled:opacity-50" disabled={busy} onClick={() => setOn(!on)}>
          {on ? "Stop" : "Start camera"}
        </button>
        <select className="rounded bg-slate-800 px-2 py-1.5" value={backend} disabled={busy} onChange={(e) => setBackend(e.target.value as Backend)}>
          {hasGpu && <option value="webgpu">WebGPU</option>}
          <option value="wasm">WebAssembly</option>
        </select>
        <select className="rounded bg-slate-800 px-2 py-1.5" value={variant} disabled={busy} onChange={(e) => setVariant(e.target.value)}>
          {["fp32", "fp16", "fp16-head32"].filter((v) => v in config.models).map((v) => <option key={v}>{v}</option>)}
        </select>
        {camera.devices.length > 1 && (
          <select className="max-w-48 rounded bg-slate-800 px-2 py-1.5" value={deviceId ?? ""} disabled={busy} onChange={(e) => setDeviceId(e.target.value || null)}>
            <option value="">back camera</option>
            {camera.devices.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label}</option>)}
          </select>
        )}
        {detector && (
          <>
            <select className="rounded bg-slate-800 px-2 py-1.5" value={kind} disabled={busy} onChange={(e) => setKind(e.target.value as Kind)}>
              <option value="scan">scan, 60 s</option>
              <option value="soak">soak, 5 min</option>
            </select>
            {busy ? (
              <button className="rounded bg-rose-600 px-3 py-1.5 font-medium text-white" disabled={recording?.phase !== "recording"}
                onClick={() => (stopAsked.current = true)}>Stop and post</button>
            ) : (
              <button className="rounded bg-emerald-600 px-3 py-1.5 font-medium text-white" onClick={record}>Record</button>
            )}
          </>
        )}
      </div>

      <div className="relative w-full overflow-hidden rounded bg-black" style={{ height: "70vh" }}>
        <video ref={video} className="h-full w-full object-contain" playsInline muted />
        <LiveBoxes boxes={view.boxes} frameWidth={view.frame[0]} frameHeight={view.frame[1]} classes={config.classes}
          dim={view.motion !== "still" && view.motion !== "off"} />
        {recording && (
          <div className="absolute inset-x-0 top-0 bg-black/60 p-2 text-center text-base font-semibold text-white">
            {recording.phase === "countdown" ? `${kind} starts in ${recording.left} s`
              : recording.phase === "posting" ? "posting the record…"
              : `${kind}: ${recording.left} s left · ${KINDS[kind].cue(recording.elapsed)}`}
          </div>
        )}
      </div>

      <p className="font-mono text-xs text-slate-300">
        {backendName(backend)} · {variant} · {SQUARE}×{SQUARE}
        {last ? ` · ${last.ms.total.toFixed(0)} ms a detection (model ${last.ms.run.toFixed(0)}) · ${view.rate.toFixed(1)}/s` : ""}
        {` · ${motion}`}
        {view.gateMs !== null ? ` · gate ${view.gateMs.toFixed(2)} ms` : ""}
        {last ? ` · ${view.boxes.length} boxes` : ""}
        {s ? ` · camera ${s.width}×${s.height} ${s.frameRate ? `${Math.round(s.frameRate)} fps` : ""}` : ""}
      </p>
      {config.gate === null && <p className="text-sm text-amber-300">No gate yet: record Motion first; until then detection runs continuously.</p>}
      {longSide > 0 && longSide < SQUARE && (
        <p className="text-sm text-amber-400">The camera gives {s?.width}×{s?.height}: under 1,280 px on the long side, the model sees print smaller than it learned.</p>
      )}
      {posted && <p className="text-xs text-slate-400">record: {posted}</p>}
      {status && <p className="text-sm text-amber-300">{status}</p>}
      {camera.error && <p className="text-sm text-red-400">{camera.error}</p>}
      {info && (
        <p className="text-xs text-slate-400">
          ONNX Runtime {info.ortVersion} · threads {info.threads} · {info.crossOriginIsolated ? "isolated" : "not isolated"} ·
          model {(info.modelBytes / 1e6).toFixed(1)} MB in {info.downloadMs.toFixed(0)} ms · built in {info.buildMs.toFixed(0)} ms ·
          warm-up {info.warmupMs.map((m) => m.toFixed(0)).join(", ")} ms
          {info.gpu ? ` · GPU ${info.gpu.vendor} ${info.gpu.architecture} ${info.gpu.description} · shader-f16 ${info.gpu.shaderF16 ? "yes" : "no"}` : ""}
          {camera.label ? ` · ${camera.label} · focus ${camera.focus}` : ""}
          {config.gate?.on ? ` · gate T ${config.gate.t!.toFixed(2)}, every ${config.gate.every}` : ""} · wake lock {wakeLock}
        </p>
      )}
    </div>
  );
}
