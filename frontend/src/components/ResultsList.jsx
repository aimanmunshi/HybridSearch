import { ResultCard } from "./ResultCard";
import { SkeletonCard } from "./SkeletonCard";

export function ResultsList({ status, results, error, hasSearched }) {
  if (status === "loading") {
    return (
      <ul className="mt-6 space-y-3">
        {Array.from({ length: 4 }, (_, i) => (
          <SkeletonCard key={i} />
        ))}
      </ul>
    );
  }

  if (status === "error") {
    return (
      <div className="mt-6 rounded-xl border border-chili/40 bg-chili/10 p-6 text-center font-sans text-sm text-chalk">
        Something boiled over: {error}
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="mt-10 text-center font-display text-2xl text-chalk-dim">
        Type a craving, not a keyword. Try "something spicy to warm up with".
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="mt-10 text-center font-sans text-sm text-chalk-dim">
        Nothing on the menu matched that. Try describing the dish differently.
      </div>
    );
  }

  const scores = results.map((r) => r.score);
  const scoreRange = { min: Math.min(...scores), max: Math.max(...scores) };

  return (
    <ul className="mt-6 space-y-3">
      {results.map((result) => (
        <ResultCard key={result.recipe_id} result={result} scoreRange={scoreRange} />
      ))}
    </ul>
  );
}
