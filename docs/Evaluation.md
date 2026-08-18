# Evaluation

Every figure this project reported before this harness existed was a count:
how many nodes, how many edges, how many cross-modal links. Counts cannot
distinguish *more* edges from *better* edges. This harness measures whether the
graph supports predicting associations that were not known when the training
data was assembled.

The headline result is negative, and worth stating plainly: **the knowledge
graph as currently built does not support topological link prediction.** The
harness is what makes that statement possible.

## Running it

```bash
make evaluate                    # cutoff 2016, popularity controlled
make evaluate CUTOFF=2014
make evaluate-years              # choose a cutoff from the year distribution
```

Or directly, for the full option set:

```bash
python scripts/evaluate_link_prediction.py --cutoff 2016 --degree-matched
```

| Flag | Effect |
|---|---|
| `--cutoff YEAR` | Papers published before `YEAR` are training data |
| `--degree-matched` | Draw negatives from the positive's degree bucket |
| `--no-type-matching` | Draw negatives ignoring entity type (easier; for comparison) |
| `--negatives N` | Negatives sampled per positive (default 10) |
| `--list-years` | Print the year distribution and exit |
| `--output PATH` | Write the report as JSON |

## Why temporal, not random

A random split leaks. An association curated in 2005 and one curated in 2021
are equally likely to land in the test set, so a model scores well by
recognising co-occurrence patterns it has already seen. Splitting on the
publication year of the supporting paper asks the question that matters: given
what was published before year Y, does the graph predict what came after?

CIVIC's `last_review_date` looks like the obvious field and is **not usable**
— 4171 of 4254 rows share a 2023 bulk re-review timestamp, which records when
a curator last looked at a record, not when the finding was published. The year
is instead parsed from the citation string ("Levine et al., 2005"), which
succeeds on 4253 of 4254 rows and spans 1988–2023.

## What the split protects against

Three things silently inflate temporal-holdout numbers. Each is handled
explicitly and counted in the report.

**Re-assertion.** A pair first published in 2010 and cited again in 2020 is not
a 2020 discovery. Keyed on the *first* year a pair appears, so it stays in
training. At the 2016 cutoff, **495 pairs** were re-asserted after the cutoff
having first appeared before it. Treating "any evidence after the cutoff" as
test data would have scored all 495 as correct predictions of things the model
had already been shown.

**Backbone overlap.** Gene→variant edges carry no publication date — a variant
belonging to a gene is not a discovery that happens in a year. They are always
in the training graph, and any test pair that duplicates one is dropped.

**Cold start.** A pair with an endpoint absent from the training graph cannot
be scored by any topological method. These are excluded and reported
separately rather than counted as failures. At the 2016 cutoff, 366 of 1570.

## Negative sampling

Uniform negatives make the task trivial. Most random pairs are type
incompatible — a phenotype and a gene — so a predictor scores well by learning
which *type pairs* are plausible rather than which *associations* are real.
Negatives therefore match the endpoint types of the positive they stand
against.

`--degree-matched` goes further and draws each negative endpoint from the same
log-scale degree bucket as its positive. This matters more than it sounds:

| Predictor | Type-matched only | Degree-matched |
|---|---|---|
| preferential_attachment | **0.725** | 0.512 |
| adamic_adar | 0.562 | 0.543 |
| common_neighbors | 0.561 | 0.542 |
| jaccard | 0.560 | 0.544 |
| random | 0.498 | 0.498 |

Preferential attachment multiplies degrees and ignores shared structure
entirely. Its AUC of 0.725 collapsing to 0.512 under degree matching shows the
whole apparent signal was popularity: the test set is dominated by well-studied
diseases (`DOID:162` "cancer" has training degree 171 and appears in 56 test
edges). Preferential attachment is included as a control precisely so this is
visible — when it wins, the evaluation is measuring fame, not prediction.

## Results

Temporal holdout at 2016, degree-matched, 10 negatives per positive:

```
predictor                      AUC      AP     H@1     H@5    H@10     MRR
--------------------------------------------------------------------------
jaccard                      0.544   0.115   0.000   0.000   0.001   0.001
adamic_adar                  0.543   0.106   0.000   0.000   0.001   0.001
common_neighbors             0.542   0.105   0.000   0.000   0.001   0.001
preferential_attachment      0.511   0.097   0.000   0.001   0.003   0.001
random                       0.498   0.092   0.000   0.000   0.002   0.001
```

Stable across cutoffs — Adamic-Adar 0.536 (2012), 0.547 (2014), 0.543 (2016),
0.558 (2018), against a random floor of 0.504–0.510.

## Why the numbers are this low

**84.6% of test pairs share no neighbour with each other in the training
graph.** Adamic-Adar, common neighbours and Jaccard all return exactly 0.0 for
such a pair. They are not ranking those pairs badly — they cannot rank them at
all. That is why Hits@1 is 0.000 across every predictor while AUC still reads
above 0.5: AUC is computed over a distribution dominated by ties, and the head
of the ranking, which is what anyone would actually look at, is empty.

The training graph is 2613 nodes and 6769 edges with a median degree of 3 and
average clustering of 0.185. It is close to bipartite in practice — variants
attach to diseases and therapies with little of the triangle structure that
shared-neighbour prediction requires.

The harness reports this itself as `structural_coverage` and emits a warning
below 50%, because a metric that is undefined for most of the test set should
say so rather than be read as a weak result.

## What this implies

Adding a GNN will not fix this on its own. Message passing propagates over the
same sparse topology; if 85% of target pairs have no shared neighbour, there
is no path of length two to propagate along. The plausible directions are:

1. **Densify the graph.** More relations per node — more literature, more
   sources — is the direct attack on structural coverage.
2. **Use features, not topology.** Node attributes (name and description
   embeddings, sequence or structure features) can score pairs that share no
   neighbours. This is where a learned model would earn its complexity.
3. **Reframe the task.** Ranking therapies for a given variant is a smaller,
   better-posed problem than open link prediction, and the type-annotated
   graph now supports it.

Any of these is testable with this harness, which is the point of it.

## Adding a predictor

Implement `score`, and the harness handles the rest:

```python
from litkg.evaluation import LinkPredictor, evaluate_baselines, build_temporal_split

class MyPredictor(LinkPredictor):
    name = "mine"

    def fit(self, graph):
        self.graph = graph      # or train a model here
        return self

    def score(self, u, v):
        return ...

split = build_temporal_split(dated_edges, cutoff_year=2016, backbone_edges=backbone)
report = evaluate_baselines(
    split, node_types=types, predictors=[MyPredictor()], degree_matched=True
)
print(report.format_table())
```

Report any new model against these baselines with degree matching on. A model
that beats random but not preferential attachment has learned popularity.
