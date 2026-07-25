/**
 * The relevance "heat gauge" -- this project's signature element. A search
 * score is rendered the same way a spice level or oven temperature would be:
 * a thermometer filling from cool sage to chili red. It doubles as an honest
 * disclosure that the number itself isn't on a fixed scale (semantic
 * similarity, blended hybrid score, and cross-encoder logits all use
 * different ranges) -- what's comparable is *rank within this result set*,
 * which the fill height shows directly.
 */
export function ScoreGauge({ score, min, max }) {
  const pct = max > min ? ((score - min) / (max - min)) * 100 : 100;

  return (
    <div className="flex w-12 shrink-0 flex-col items-center gap-1.5">
      <div className="relative h-20 w-2 overflow-hidden rounded-full bg-board-line">
        <div className="absolute inset-x-0 bottom-0 rounded-full bg-gradient-to-t from-sage via-turmeric to-chili" style={{ height: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] tabular-nums text-chalk-dim">{score.toFixed(2)}</span>
    </div>
  );
}
