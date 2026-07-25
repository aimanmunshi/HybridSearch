export function SearchBar({ value, onChange, onSubmit, loading }) {
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit();
      }}
      className="flex gap-2"
    >
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="warm comfort food for a rainy day…"
        className="flex-1 rounded-lg border border-board-line bg-board-raised px-4 py-3 font-sans text-[15px] text-chalk placeholder:text-chalk-dim/70 focus:border-turmeric focus:outline-none focus:ring-2 focus:ring-turmeric/30"
        autoFocus
      />
      <button
        type="submit"
        disabled={loading || !value.trim()}
        className="rounded-lg bg-chili px-6 py-3 font-sans font-semibold text-chalk transition hover:bg-chili-dim disabled:cursor-not-allowed disabled:opacity-40"
      >
        {loading ? "Searching…" : "Search"}
      </button>
    </form>
  );
}
