/**
 * The live-boxes demo (docs/m2c-live-boxes.md): Live and Bench. The page is
 * served by Vite (web/), its API by scripts/live_api.py on the laptop.
 */
import { useEffect, useState } from "react";
import type { ApiConfig } from "./api";
import { apiConfig, hashParams, key } from "./api";
import { Bench } from "./pages/Bench";
import { Live } from "./pages/Live";

type Page = "live" | "bench";

export function App() {
  const [page, setPage] = useState<Page>(hashParams().get("page") === "bench" ? "bench" : "live");
  const [config, setConfig] = useState<ApiConfig | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (key()) apiConfig().then(setConfig).catch((e) => setError(String(e)));
  }, []);

  const tab = (p: Page, label: string) => (
    <button className={`rounded px-3 py-1 ${page === p ? "bg-slate-700 text-white" : "text-slate-400"}`} onClick={() => setPage(p)}>
      {label}
    </button>
  );
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-3 p-3">
      <nav className="flex items-center gap-1 text-sm">
        <span className="mr-2 font-semibold">BestBefore live</span>
        {tab("live", "Live")}
        {tab("bench", "Bench")}
      </nav>
      {!key() && <p className="text-amber-300">No key: open the link the laptop's API printed (it ends in #k=…).</p>}
      {error && <p className="text-red-400">{error}</p>}
      {config && (page === "live" ? <Live config={config} /> : <Bench config={config} />)}
    </main>
  );
}
