/**
 * Run records, posted to the API, which adds its git commit, the models'
 * hashes and the time, and writes them to results/runs/m2c/
 * (docs/m2c-live-boxes.md, "Records and scores").
 */
import { apiJson } from "./api";

export type RecordKind = "bench-parity" | "bench-speed" | "live" | "motion" | "gate-bench" | "error";

export function postRecord(kind: RecordKind, body: object): Promise<{ file: string; run: string }> {
  return apiJson(`/api/records/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** What failed, posted so the laptop sees it: on the phone nothing else is seen. */
export function postError(where: string, error: unknown, context: object = {}): void {
  postRecord("error", { where, message: String(error), stack: error instanceof Error ? error.stack : undefined,
    context, userAgent: navigator.userAgent, app: APP }).catch(() => {});
}

/** The app's own build: the commit it was built from (vite.config.ts). */
export const APP = { commit: __APP_COMMIT__, dirty: __APP_DIRTY__ };

interface UAData {
  mobile?: boolean;
  platform?: string;
  getHighEntropyValues?: (hints: string[]) => Promise<Record<string, unknown>>;
}

interface GpuAdapterLike {
  info?: Record<string, string>;
  features?: Iterable<string>;
}

/** What the record says about the device: browser, cores, memory, screen,
 *  cross-origin isolation, and WebGPU's adapter as the page sees it. */
export async function deviceInfo(): Promise<Record<string, unknown>> {
  const nav = navigator as Navigator & {
    deviceMemory?: number;
    userAgentData?: UAData;
    gpu?: { requestAdapter: () => Promise<GpuAdapterLike | null> };
  };
  let uaData: Record<string, unknown> | null = null;
  if (nav.userAgentData?.getHighEntropyValues) {
    uaData = await nav.userAgentData
      .getHighEntropyValues(["model", "platformVersion", "architecture", "fullVersionList"])
      .catch(() => null);
  }
  let webgpu: Record<string, unknown> = { available: Boolean(nav.gpu) };
  if (nav.gpu) {
    const adapter = await nav.gpu.requestAdapter().catch(() => null);
    if (adapter) {
      const features = [...(adapter.features ?? [])].sort();
      const info = adapter.info ?? {};
      webgpu = {
        available: true, vendor: info.vendor, architecture: info.architecture, device: info.device,
        description: info.description, features, shaderF16: features.includes("shader-f16"),
      };
    } else {
      webgpu = { available: true, adapter: null };
    }
  }
  return {
    userAgent: navigator.userAgent, uaData, cores: navigator.hardwareConcurrency,
    memoryGB: nav.deviceMemory ?? null, crossOriginIsolated, webgpu,
    screen: { width: screen.width, height: screen.height, dpr: devicePixelRatio },
  };
}
