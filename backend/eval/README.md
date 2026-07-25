# Evaluation harness

```bash
python -m eval.run_eval
```

Runs all 20 queries in `queries.json` through semantic-only, hybrid, and
hybrid+rerank search, computes precision@5 against hand-curated ground truth,
and writes `results/report.md` and `results/report.json`.

## Ground truth methodology

Labeling by eye ("does this look relevant?") would make the eval circular --
it's too easy to unconsciously favor whichever ranking looks more familiar.
Instead, every query's expected-result set was derived from **objective
corpus metadata** using a documented SQL filter (see the `note` field on each
entry in `queries.json`):

- **Exact-term queries** (`gochujang`, `harissa`): a substring match on the
  recipe's full text. Either the ingredient is mentioned or it isn't.
- **Category/cuisine queries** (`quick pasta dinner`, `turkish cuisine`,
  `french dessert`, ...): a single-column match against `recipes.category` or
  `recipes.cuisine` -- TheMealDB's own curated classification, not something
  this project inferred.
- **Combined queries** (`chicken curry`, `italian pasta`, ...): category or
  cuisine AND a title/tag match, for queries that name both a dish type and a
  cuisine.

### Two manual corrections

A review pass over every candidate list (reading actual titles, not just
counts) caught two mistagged community tags: **"Spicy Arrabiata Penne"** (an
Italian pasta dish) and **"Jerk chicken with rice & peas"** were both tagged
`Curry` in the source data despite not being curries. Both were manually
excluded from their respective ground-truth sets, and the exclusion is noted
in `queries.json`.

### Two queries dropped entirely

`BBQ` and `Baking` were tried as tag-based ground truth and rejected: reviewing
the actual matches showed the tags themselves are unreliable in this corpus
(`Chocolate Caramel Crispy` and `Peanut Butter Cookies` are tagged `BBQ`;
`Mediterranean Pasta Salad` -- a cold dish -- is tagged `Baking`). Rather than
build an eval query on top of noisy labels, those two were replaced with
`turkish cuisine` and `jamaican dish` (clean, single-column cuisine matches).

This is disclosed rather than hidden because it's a real property of
community-contributed data: **tags are the noisiest field in this corpus**,
which is also why `chunk.py`'s header only ever uses tags as one signal among
several, never the sole basis for anything.

## Why precision@5, not recall or NDCG

Precision@5 answers the question a user actually has: *were the first five
things I saw worth looking at?* Recall would require knowing the total
number of "relevant" recipes in the whole corpus for a fuzzy query like "warm
comfort food," which metadata-derived ground truth can't provide gradations
of. NDCG needs graded relevance (this result is a 3/5 match, that one's a
1/5) -- but the ground truth here is binary (matches the filter or doesn't),
so NDCG would just be a more complicated way of computing the same thing.
