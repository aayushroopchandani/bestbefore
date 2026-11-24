// Copies ONNX Runtime Web's WebAssembly runtime into public/ort/, which the
// app names as env.wasm.wasmPaths (docs/m2c-live-boxes.md, S2). Run by npm's
// `prepare`, so `npm ci` leaves it in place; public/ort/ isn't committed.
//
// onnxruntime-web 1.31.1's default import (the JSEP build: WebGPU and
// WebAssembly in one runtime) bundles a loader for these two files. Given
// wasmPaths, it imports the .mjs from there instead of its own copy, and the
// runtime's thread workers start from that .mjs, not from the app's worker.
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const dist = join(web, "node_modules", "onnxruntime-web", "dist");
const out = join(web, "public", "ort");
const files = ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"];

mkdirSync(out, { recursive: true });
for (const f of files) copyFileSync(join(dist, f), join(out, f));
console.log(`copied ${files.join(", ")} to public/ort/`);
