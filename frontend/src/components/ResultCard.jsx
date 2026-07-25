import { ScoreGauge } from "./ScoreGauge";

export function ResultCard({ result, scoreRange }) {
  const meta = [result.cuisine, result.category].filter(Boolean).join(" · ");

  return (
    <li className="flex gap-4 rounded-xl border border-board-line bg-board-raised p-4">
      <ScoreGauge score={result.score} min={scoreRange.min} max={scoreRange.max} />

      {result.thumbnail_url && (
        <img
          src={result.thumbnail_url}
          alt=""
          className="h-20 w-20 shrink-0 rounded-lg object-cover"
          loading="lazy"
        />
      )}

      <div className="min-w-0 flex-1">
        <h3 className="line-clamp-2 font-sans text-base font-semibold text-chalk">{result.title}</h3>
        {meta && <p className="mt-0.5 font-sans text-xs uppercase tracking-wide text-turmeric">{meta}</p>}
        <p className="mt-2 font-sans text-sm leading-relaxed text-chalk-dim">{result.snippet}</p>
        {result.tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {result.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-board-line px-2 py-0.5 font-mono text-[10px] text-chalk-dim"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </div>
    </li>
  );
}
