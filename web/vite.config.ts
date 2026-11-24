/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import basicSsl from "@vitejs/plugin-basic-ssl";
import { execSync } from "node:child_process";

/**
 * Serving the live-boxes app to the phone (docs/m2c-live-boxes.md, "Serving
 * it to the phone"), as Scanify's frontend serves its scanner.
 *
 * - HTTPS with a self-signed certificate: a browser gives the camera only
 *   to a secure page, and the phone reaches the laptop by its Wi-Fi address,
 *   not localhost. The phone warns once; the owner accepts.
 * - `/api` is proxied to the Python API (scripts/live_api.py), which listens
 *   on 127.0.0.1 only: the page and its API are one origin, so no mixed
 *   content and no CORS. The API answers only requests carrying its key.
 * - COOP and COEP make the page cross-origin isolated, which WebAssembly
 *   threads need (SharedArrayBuffer). Every response gets them, the worker
 *   scripts' included: a worker is isolated only if its own script is.
 * - Fixed ports (strictPort), so Scanify's dev servers (5173, 8000) can run
 *   beside it. The same settings for `vite preview`, which serves the built
 *   app that every measurement runs on.
 */
const headers = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};
// xfwd: the API gets the browser's address, and gives store photos only to this laptop.
const proxy = { "/api": { target: "http://127.0.0.1:8765", xfwd: true } };
const serve = { host: true, port: 5180, strictPort: true, proxy, headers };

/** The commit the app is built from, for every record it posts. */
function git(args: string): string {
  try {
    return execSync(`git ${args}`, { encoding: "utf8" }).trim();
  } catch {
    return "";
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), basicSsl()],
  // ONNX Runtime Web loads its WebAssembly from public/ort/ (env.wasm.wasmPaths),
  // so Vite mustn't pre-bundle it and move the files it expects beside it.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  worker: { format: "es" },
  define: {
    __APP_COMMIT__: JSON.stringify(git("rev-parse HEAD")),
    __APP_DIRTY__: JSON.stringify(git("status --porcelain") !== ""),
  },
  server: serve,
  preview: serve,
  test: { include: ["src/**/*.test.ts"], environment: "node" },
});
