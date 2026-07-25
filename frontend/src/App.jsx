import { useState } from "react";
import { ModeToggle } from "./components/ModeToggle";
import { ResultsList } from "./components/ResultsList";
import { SearchBar } from "./components/SearchBar";
import { search } from "./lib/api";

function App() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [rerank, setRerank] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | loading | done | error
  const [results, setResults] = useState([]);
  const [meta, setMeta] = useState(null); // { tookMs, mode }
  const [error, setError] = useState(null);

  async function runSearch() {
    const trimmed = query.trim();
    if (!trimmed) return;

    setStatus("loading");
    setError(null);
    try {
      const data = await search(mode, trimmed, { rerank });
      setResults(data.results);
      setMeta({ tookMs: data.took_ms, mode: data.mode });
      setStatus("done");
    } catch (err) {
      setError(err.message);
      setStatus("error");
    }
  }

  return (
    <div className="min-h-screen bg-board">
      <div className="mx-auto max-w-2xl px-6 py-12">
        <header className="mb-8 text-center">
          <p className="font-sans text-xs uppercase tracking-[0.2em] text-chalk-dim">
            today's specials, found by meaning
          </p>
          <h1 className="mt-2 font-display text-5xl font-bold text-chalk">
            What are you craving?
          </h1>
        </header>

        <SearchBar value={query} onChange={setQuery} onSubmit={runSearch} loading={status === "loading"} />

        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <ModeToggle mode={mode} onChange={setMode} rerank={rerank} onRerankChange={setRerank} />
          {meta && status === "done" && (
            <p className="font-mono text-xs text-chalk-dim">
              {results.length} results · {meta.mode} · {meta.tookMs.toFixed(0)}ms
            </p>
          )}
        </div>

        <ResultsList status={status} results={results} error={error} hasSearched={status !== "idle"} />
      </div>
    </div>
  );
}

export default App;
