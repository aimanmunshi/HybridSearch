const MODES = [
  { value: "semantic", label: "Semantic", hint: "meaning only" },
  { value: "hybrid", label: "Hybrid", hint: "meaning + keywords" },
];

export function ModeToggle({ mode, onChange, rerank, onRerankChange }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <div className="inline-flex rounded-lg border border-board-line bg-board-raised p-1">
        {MODES.map((m) => (
          <button
            key={m.value}
            type="button"
            onClick={() => onChange(m.value)}
            aria-pressed={mode === m.value}
            className={`rounded-md px-4 py-1.5 font-sans text-sm font-medium transition ${
              mode === m.value
                ? "bg-chili text-chalk shadow-sm"
                : "text-chalk-dim hover:text-chalk"
            }`}
          >
            {m.label}
            <span className="ml-1.5 hidden font-normal opacity-70 sm:inline">({m.hint})</span>
          </button>
        ))}
      </div>

      <label
        className={`flex items-center gap-2 font-sans text-sm transition ${
          mode === "hybrid" ? "text-chalk-dim" : "cursor-not-allowed text-chalk-dim/40"
        }`}
      >
        <input
          type="checkbox"
          checked={rerank}
          disabled={mode !== "hybrid"}
          onChange={(e) => onRerankChange(e.target.checked)}
          className="size-4 accent-chili disabled:opacity-40"
        />
        rerank (cross-encoder)
      </label>
    </div>
  );
}
