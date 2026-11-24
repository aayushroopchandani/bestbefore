/**
 * Live (S2's first version; docs/m2c-live-boxes.md, "The app"): the camera,
 * one detection at a time on the newest frame, the boxes drawn. No
 * stillness gate, rest rule or tracker yet (S3): when a detection ends, the
 * next starts on the frame then on screen, and the frames between are
 * never seen by the model.
 */
import { useEffect, useRef, useState } from "react";
import type { ApiConfig } from "../api";
import { keyHeaders } from "../api";
import { Detector, backendName } from "../detector/client";
import type { Backend, DetectResult } from "../detector/types";
import { LiveBoxes } from "../live/LiveBoxes";
import { useCamera } from "../live/useCamera";
import { APP, deviceInfo, postError, postRecord } from "../records";

const SQUARE = 1280; // R1 kept square padding: every frame's input is 1,280 x 1,280

export function Live({ config }: { config: ApiConfig }) {
  const hasGpu = "gpu" in navigator;
  const [backend, setBackend] = useState<Backend>(hasGpu ? "webgpu" : "wasm");
  const [variant, setVariant] = useState("fp32");
  const [on, setOn] = useState(false);
  const [deviceId, setDeviceId] = useState<string | null>(null);
  const camera = useCamera(on, deviceId);
  const video = useRef<HTMLVideoElement>(null);
  const [detector, setDetector] = useState<Detector | null>(null);
  const [status, setStatus] = useState("");
  const [result, setResult] = useState<(DetectResult & { bitmapMs: number }) | null>(null);
  const [rate, setRate] = useState(0);
  const recent = useRef<{ at: number; ms: DetectResult["ms"]; bitmapMs: number; boxes: number; frame: [number, number] }[]>([]);
  const [saved, setSaved] = useState("");

  // S2's numbers from the phone, until S3's live records: the camera, and the last detections' times.
  const save = async () => {
    try {
      const { file } = await postRecord("live", {
        backend, variant, app: APP, device: await deviceInfo(), detector: detector?.info ?? null,
        camera: { label: camera.label, settings: camera.settings, focus: camera.focus, cameras: camera.devices.length },
        rate, detections: recent.current,
      });
      setSaved(file);
      setTimeout(() => setSaved(""), 3000);
    } catch (e) {
      setSaved(String(e));
    }
  };

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
      setResult(null);
    };
  }, [on, backend, variant, config]);

  // One detection at a time, each on the newest frame.
  useEffect(() => {
    if (!detector) return;
    let stop = false;
    const ends: number[] = [];
    (async () => {
      while (!stop) {
        const v = video.current;
        if (!v || v.readyState < 2 || !v.videoWidth) {
          await new Promise((r) => setTimeout(r, 50));
          continue;
        }
        const t0 = performance.now();
        const frame = await createImageBitmap(v);
        const bitmapMs = performance.now() - t0;
        try {
          const r = await detector.detect(frame, config.threshold);
          if (stop) break;
          const now = performance.now();
          ends.push(now);
          while (ends.length > 1 && now - ends[0] > 3000) ends.shift();
          setRate(ends.length > 1 ? ((ends.length - 1) * 1000) / (now - ends[0]) : 0);
          setResult({ ...r, bitmapMs });
          recent.current = [...recent.current.slice(-49),
            { at: Math.round(now), ms: r.ms, bitmapMs, boxes: r.boxes.length, frame: [r.frameWidth, r.frameHeight] }];
        } catch (e) {
          if (!stop && !detector.closed) {
            setStatus(`detection: ${String(e)}`);
            postError("live: detection", e, { backend: detector.config.backend });
          }
          break;
        }
      }
    })();
    return () => {
      stop = true;
    };
  }, [detector, config.threshold]);

  const s = camera.settings;
  const longSide = s?.width && s?.height ? Math.max(s.width, s.height) : 0;
  const info = detector?.info;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <button className="rounded bg-cyan-600 px-3 py-1.5 font-medium text-white" onClick={() => setOn(!on)}>
          {on ? "Stop" : "Start camera"}
        </button>
        <select className="rounded bg-slate-800 px-2 py-1.5" value={backend} onChange={(e) => setBackend(e.target.value as Backend)}>
          {hasGpu && <option value="webgpu">WebGPU</option>}
          <option value="wasm">WebAssembly</option>
        </select>
        <select className="rounded bg-slate-800 px-2 py-1.5" value={variant} onChange={(e) => setVariant(e.target.value)}>
          {["fp32", "fp16", "fp16-head32"].filter((v) => v in config.models).map((v) => <option key={v}>{v}</option>)}
        </select>
        {detector && (
          <button className="rounded bg-slate-700 px-3 py-1.5" onClick={save}>{saved ? "Saved ✓" : "Save numbers"}</button>
        )}
        {camera.devices.length > 1 && (
          <select className="max-w-48 rounded bg-slate-800 px-2 py-1.5" value={deviceId ?? ""} onChange={(e) => setDeviceId(e.target.value || null)}>
            <option value="">back camera</option>
            {camera.devices.map((d) => <option key={d.deviceId} value={d.deviceId}>{d.label}</option>)}
          </select>
        )}
      </div>

      <div className="relative w-full overflow-hidden rounded bg-black" style={{ height: "70vh" }}>
        <video ref={video} className="h-full w-full object-contain" playsInline muted />
        {result && <LiveBoxes boxes={result.boxes} frameWidth={result.frameWidth} frameHeight={result.frameHeight} classes={config.classes} />}
      </div>

      <p className="font-mono text-xs text-slate-300">
        {backendName(backend)} · {variant} · {SQUARE}×{SQUARE}
        {result ? ` · ${result.ms.total.toFixed(0)} ms a detection (model ${result.ms.run.toFixed(0)}) · ${rate.toFixed(1)}/s · ${result.boxes.length} boxes` : ""}
        {s ? ` · camera ${s.width}×${s.height} ${s.frameRate ? `${Math.round(s.frameRate)} fps` : ""}` : ""}
      </p>
      {longSide > 0 && longSide < SQUARE && (
        <p className="text-sm text-amber-400">The camera gives {s?.width}×{s?.height}: under 1,280 px on the long side, the model sees print smaller than it learned.</p>
      )}
      {saved && <p className="text-xs text-slate-400">saved: {saved}</p>}
      {status && <p className="text-sm text-amber-300">{status}</p>}
      {camera.error && <p className="text-sm text-red-400">{camera.error}</p>}
      {info && (
        <p className="text-xs text-slate-400">
          ONNX Runtime {info.ortVersion} · threads {info.threads} · {info.crossOriginIsolated ? "isolated" : "not isolated"} ·
          model {(info.modelBytes / 1e6).toFixed(1)} MB in {info.downloadMs.toFixed(0)} ms · built in {info.buildMs.toFixed(0)} ms ·
          warm-up {info.warmupMs.map((m) => m.toFixed(0)).join(", ")} ms
          {info.gpu ? ` · GPU ${info.gpu.vendor} ${info.gpu.architecture} ${info.gpu.description} · shader-f16 ${info.gpu.shaderF16 ? "yes" : "no"}` : ""}
          {camera.label ? ` · ${camera.label} · focus ${camera.focus}` : ""}
        </p>
      )}
    </div>
  );
}
