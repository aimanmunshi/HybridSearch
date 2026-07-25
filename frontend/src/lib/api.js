const BASE = "/api/search";

/**
 * @param {"semantic"|"hybrid"} mode
 * @param {string} query
 * @param {{topK?: number, rerank?: boolean}} [options]
 */
export async function search(mode, query, { topK = 10, rerank = false } = {}) {
  const params = new URLSearchParams({ q: query, top_k: String(topK) });
  if (mode === "hybrid" && rerank) params.set("rerank", "true");

  const response = await fetch(`${BASE}/${mode}?${params}`);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg).join("; ")
      : detail || `Search failed (${response.status})`;
    throw new Error(message);
  }
  return response.json();
}
