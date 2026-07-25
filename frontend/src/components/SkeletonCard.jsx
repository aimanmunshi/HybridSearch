export function SkeletonCard() {
  return (
    <li className="flex gap-4 rounded-xl border border-board-line bg-board-raised p-4 animate-pulse">
      <div className="h-20 w-2 shrink-0 rounded-full bg-board-line" />
      <div className="h-20 w-20 shrink-0 rounded-lg bg-board-line" />
      <div className="min-w-0 flex-1 space-y-2 py-1">
        <div className="h-4 w-2/3 rounded bg-board-line" />
        <div className="h-3 w-1/3 rounded bg-board-line" />
        <div className="h-3 w-full rounded bg-board-line" />
        <div className="h-3 w-4/5 rounded bg-board-line" />
      </div>
    </li>
  );
}
