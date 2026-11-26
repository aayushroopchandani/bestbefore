/**
 * The API (scripts/live_api.py), through Vite's /api proxy, and its key.
 *
 * The laptop prints a link https://<laptop>:5180/#k=<key>. What follows the
 * `#` never leaves the browser in a request, so the key is in no request
 * line or log; the app moves it into session storage (gone when the tab
 * closes), takes it out of the address bar, and sends it as a header.
 * The other `#` parameters (page=bench&set=...) stay for the pages to read.
 */
const KEY_HEADER = "X-Live-Key";
const STORAGE = "bestbefore-live-key";

export function hashParams(): URLSearchParams {
  return new URLSearchParams(location.hash.slice(1));
}

/** Moves a `k` in the URL's fragment into session storage. */
export function takeKey(): void {
  const params = hashParams();
  const key = params.get("k");
  if (!key) return;
  try {
    sessionStorage.setItem(STORAGE, key);
  } catch {
    // storage refused (a private tab): the key stays in memory only
    memoryKey = key;
  }
  params.delete("k");
  const rest = params.toString();
  history.replaceState(null, "", `${location.pathname}${location.search}${rest ? `#${rest}` : ""}`);
}

let memoryKey: string | null = null;

export function key(): string | null {
  try {
    return sessionStorage.getItem(STORAGE) ?? memoryKey;
  } catch {
    return memoryKey;
  }
}

export function keyHeaders(): Record<string, string> {
  const k = key();
  return k ? { [KEY_HEADER]: k } : {};
}

export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  for (const [k, v] of Object.entries(keyHeaders())) headers.set(k, v);
  const response = await fetch(path, { ...init, headers });
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${path}: HTTP ${response.status} ${text.slice(0, 200)}`);
  }
  return response;
}

export async function apiJson<T>(path: string, init?: RequestInit): Promise<T> {
  return (await apiFetch(path, init)).json() as Promise<T>;
}

export interface ModelFile {
  sha256: string;
  bytes: number;
  url: string;
}

/** The stillness gate's T, by rule G on a Motion record (live.gate); off when still and moving overlapped. */
export interface GateConfig {
  on: boolean;
  t: number | null;
  every: number; // the gate runs on every Nth frame
  still_p95: number;
  moving_p5: number;
  still: number;
  moving: number;
  record: string;
}

export interface ApiConfig {
  threshold: number; // M2b's threshold_store, unrounded
  gate: GateConfig | null; // null until a Motion record is pinned
  classes: string[];
  imgsz: number;
  sets: string[]; // the sets this browser may bench
  models: Record<string, ModelFile>; // by variant: fp32, fp16, fp16-head32, ...
  run: string; // the API's run folder
}

export function apiConfig(): Promise<ApiConfig> {
  return apiJson<ApiConfig>("/api/config");
}
