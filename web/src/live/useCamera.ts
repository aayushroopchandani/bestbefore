/**
 * The camera (docs/m2c-live-boxes.md, "The app"): the back camera
 * (facingMode environment), 1920 x 1080 at 30 fps asked for, whatever
 * arrives reported; a list of cameras, because some phones open the
 * ultra-wide one first (smaller print); continuous focus where the browser
 * offers it (Chrome on Android; Safari focuses by itself).
 */
import { useEffect, useState } from "react";

export interface Camera {
  stream: MediaStream | null;
  error: string | null;
  devices: { deviceId: string; label: string }[];
  label: string;
  settings: MediaTrackSettings | null;
  focus: string; // "continuous" if set, else what the browser offers or "not offered"
}

const ASK = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };

export function useCamera(on: boolean, deviceId: string | null): Camera {
  const [camera, setCamera] = useState<Camera>({
    stream: null, error: null, devices: [], label: "", settings: null, focus: "",
  });

  useEffect(() => {
    if (!on) return;
    let stream: MediaStream | null = null;
    let cancelled = false;
    (async () => {
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("no camera API: the page needs HTTPS");
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: deviceId ? { deviceId: { exact: deviceId }, ...ASK } : { facingMode: { ideal: "environment" }, ...ASK },
        });
        if (cancelled) return;
        const track = stream.getVideoTracks()[0];
        const focus = await continuousFocus(track);
        const devices = (await navigator.mediaDevices.enumerateDevices())
          .filter((d) => d.kind === "videoinput")
          .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `camera ${i + 1}` }));
        if (cancelled) return;
        setCamera({ stream, error: null, devices, label: track.label, settings: track.getSettings(), focus });
      } catch (e) {
        if (!cancelled) setCamera((c) => ({ ...c, stream: null, error: String(e) }));
      }
    })();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [on, deviceId]);

  return camera;
}

/** Asks for continuous focus where the track offers it. */
async function continuousFocus(track: MediaStreamTrack): Promise<string> {
  const caps = (track.getCapabilities?.() ?? {}) as { focusMode?: string[] };
  if (!caps.focusMode?.length) return "not offered";
  if (!caps.focusMode.includes("continuous")) return `offered: ${caps.focusMode.join(", ")}`;
  try {
    await track.applyConstraints({ advanced: [{ focusMode: "continuous" } as MediaTrackConstraintSet] });
    return "continuous";
  } catch (e) {
    return `refused: ${String(e)}`;
  }
}
