/**
 * The page around the loop (docs/m2c-live-boxes.md, "The frame loop"):
 * whether it is visible (the loop stops while hidden), and a screen wake
 * lock while scanning, so the phone doesn't dim and stop the camera.
 */
import { useEffect, useState } from "react";

export function useVisible(): boolean {
  const [visible, setVisible] = useState(document.visibilityState === "visible");
  useEffect(() => {
    const changed = () => setVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", changed);
    return () => document.removeEventListener("visibilitychange", changed);
  }, []);
  return visible;
}

type WakeLockSentinelLike = { release(): Promise<void>; addEventListener(t: "release", f: () => void): void };
type WakeLockNavigator = Navigator & { wakeLock?: { request(type: "screen"): Promise<WakeLockSentinelLike> } };

/** Holds the screen on while `on` and the page is visible (the browser drops the
 *  lock when the page is hidden, so it is asked for again on return). Says what
 *  happened: held, released, refused (why) or unsupported. */
export function useWakeLock(on: boolean): string {
  const visible = useVisible();
  const [state, setState] = useState("off");
  useEffect(() => {
    if (!on || !visible) return;
    const wakeLock = (navigator as WakeLockNavigator).wakeLock;
    if (!wakeLock) {
      setState("unsupported");
      return;
    }
    let lock: WakeLockSentinelLike | null = null;
    let cancelled = false;
    wakeLock.request("screen").then((l) => {
      if (cancelled) {
        void l.release();
        return;
      }
      lock = l;
      setState("held");
      l.addEventListener("release", () => setState("released"));
    }).catch((e: unknown) => setState(`refused: ${String(e)}`));
    return () => {
      cancelled = true;
      void lock?.release();
    };
  }, [on, visible]);
  return state;
}
