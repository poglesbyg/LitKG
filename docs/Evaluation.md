# Evaluation

> **Data provenance.** All figures below are from the CIVIC **01-Aug-2026**
> release (4878 evidence rows, 1992 variants, 973 features). Numbers reported
> before this release came from 01-Feb-2024 and differ; the release is recorded
> in `data/external/civic/RELEASE`. Change it with
> `LITKG_CIVIC_RELEASE=nightly` (or another dated release) and re-run — the
> default is pinned so a regression cannot be confused with a data update.


Every figure this project reported before this harness existed was a count:
how many nodes, how many edges, how many cross-modal links. Counts cannot
distinguish *more* edges from *better* edges. This harness measures whether the
graph supports predicting associations that were not known when the training
data was assembled.

The first result looked negative and was, in part, a measurement artefact.
Corrected: **the graph does carry structural signal — AUC 0.692 — but only for
predictors matched to its topology.** The initial reading of "too sparse for
link prediction" came from using shared-neighbour methods that are undefined on
this graph by construction. The harness is what made both the wrong conclusion
and its correction visible.

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
l3_paths                     0.692   0.204   0.002   0.002   0.016   0.005
jaccard                      0.544   0.117   0.000   0.001   0.002   0.002
adamic_adar                  0.543   0.107   0.000   0.000   0.001   0.001
common_neighbors             0.543   0.106   0.000   0.000   0.001   0.001
preferential_attachment      0.511   0.097   0.000   0.001   0.003   0.001
random                       0.498   0.092   0.000   0.000   0.002   0.001
```

## Why shared-neighbour methods fail here

**The graph is strictly multipartite: 0 of its 6769 edges join two nodes of the
same type, and 100% of held-out pairs are cross-type** (mutation–disease,
drug–mutation, disease–drug, mutation–phenotype).

In a multipartite graph, two nodes of different types can share a neighbour
only through some third type adjacent to both. That is rare, so Adamic-Adar,
common neighbours and Jaccard return near-zero for 84.6% of test pairs. They
are not measuring a weak graph — they are undefined on it.

Cross-type nodes in such a graph meet at *odd* distance. Of the uncovered test
pairs, most sit at distance 3 and only 14% are unreachable at all: the
connectivity is there, one hop beyond what a length-2 method can see.

`l3_paths` counts length-3 paths, normalising by the degrees of both
intermediates so hub routes count for less. Same graph, same split, same
negatives — AUC 0.543 → 0.692, average precision 0.107 → 0.204. The gain came
from matching the predictor to the topology, not from new data.

This is why `preferential_attachment` is kept as a control and why the harness
now reports `same_type_edge_ratio` and warns when the graph is multipartite. A
harness that only reported Adamic-Adar would have said this graph was useless.

## Does more data help?

Subsampling the training edges, holding the test set fixed, with `l3_paths`:

| Training edges | AUC | AP | Hits@10 |
|---|---|---|---|
| 2967 (25%) | 0.588 | 0.153 | 0.000 |
| 4235 (50%) | 0.653 | 0.180 | 0.000 |
| 5503 (75%) | 0.683 | 0.201 | 0.001 |
| 6772 (100%) | 0.689 | 0.207 | 0.022 |

AUC is decelerating hard — +0.065, +0.030, +0.006 across the quartiles — so
more edges of the same kind buy progressively less discrimination. But
**Hits@10 is still climbing steeply** at full data, which is the metric that
describes the top of the ranking anyone would actually read.

Read together: more data is not the fastest route to better AUC, but the
ranking head is still data-limited. Predictor choice was worth +0.149 AUC for
free; the next tranche of data is worth considerably less than that.

## What this implies

A GNN is now worth trying, which the earlier reading wrongly ruled out. A
two-layer network aggregates over a 2-hop neighbourhood, which is exactly the
length-3 reach that works here — `l3_paths` is close to what an untrained
2-layer GNN computes, so 0.692 is the number a trained model has to beat to
justify itself.

In rough order of expected return per unit of effort:

1. **Beat 0.692 with a trained model.** The baseline is now honest and the
   target is concrete. A GNN that lands at 0.65 is not "promising" — it is
   losing to arithmetic.
2. **Use node features.** Nothing so far uses entity names, descriptions or
   sequence data. Features can score the 14% of pairs that are topologically
   unreachable, which no amount of path counting will reach.
3. **Densify where the head is thin.** Hits@10 is still data-limited even as
   AUC saturates, so more edges should be judged on ranking quality, not AUC.
4. **Reframe to a narrower task.** Ranking therapies for a given variant is
   better posed than open link prediction and the typed graph supports it.

## Which cutoff to use

The 01-Aug-2026 release carries citations through 2025, which makes later
cutoffs viable. They are not equivalent problems:

| cutoff | train pairs | test pairs | weighted_l3 AUC | hybrid AUC | hybrid H@100 |
|---|---|---|---|---|---|
| 2016 | 4913 | 1388 | 0.693 [0.679, 0.708] | 0.748 ± 0.009 | 0.073 |
| 2020 | 6194 | 513 | 0.768 [0.747, 0.790] | **0.791 ± 0.008** | **0.131** |

**2020 scores higher because it is an easier problem, not because the method
improved.** It trains on 26% more pairs and predicts a smaller, denser test set.
Quote the cutoff with any number, and compare methods only at the same cutoff.

2016 remains the reference point for method comparisons in this document,
because every earlier result was measured there. 2020 is the more realistic
setting if the question is "what would this surface today".

## Trained models

`python scripts/train_link_prediction.py --cutoff 2016 --seeds 5`

Two learned predictors live in `litkg.phase2.link_prediction`, scored through
this same harness so the comparison is like for like.

| predictor | AUC | AUC 95% CI | AP | H@100 | MRR | MRR 95% CI |
|---|---|---|---|---|---|---|
| **hybrid** (GNN + weighted L3 + text) | **0.750** | [0.736, 0.763] | **0.300** | **0.105** | 0.0171 | [0.0117, 0.0244] |
| text_only | 0.581 | [0.566, 0.597] | 0.117 | 0.012 | 0.0012 | [0.0007, 0.0019] |
| weighted_l3 | 0.698 | [0.682, 0.713] | 0.231 | 0.053 | 0.0097 | [0.0065, 0.0133] |
| gnn alone | 0.697 | [0.685, 0.709] | 0.170 | 0.040 | 0.0031 | [0.0016, 0.0056] |
| l3_paths | 0.692 | [0.677, 0.707] | 0.204 | 0.042 | 0.0044 | [0.0031, 0.0061] |
| adamic_adar | 0.540 | [0.531, 0.551] | 0.105 | 0.019 | 0.0010 | [0.0007, 0.0014] |
| random | 0.498 | [0.482, 0.514] | 0.092 | 0.004 | 0.0008 | [0.0004, 0.0012] |

Intervals are 95% bootstrap over positives. **Read them before comparing
anything**: on this data the MRR interval is wider than the gap between most
configurations.

What this table supports:

- **The hybrid beats every structural baseline on AUC.** Its interval
  [0.729, 0.755] is disjoint from L3's [0.677, 0.707]. This holds across 5
  seeds (0.743 ± 0.010) and is the one large, solid result.
- **It also leads on AP and Hits@100**, which are the discriminating metrics
  here.
- **The GNN alone does not beat L3** — the intervals sit on top of each other,
  and across seeds it is unstable (0.670 ± 0.089, one seed collapsed to 0.512).

What it does **not** support:

- Fine-grained MRR comparisons among the leading predictors. Hybrid
  [0.0075, 0.0163] and weighted_l3 [0.0065, 0.0133] overlap heavily.
- `weighted_l3` beating `l3_paths` on AUC: 0.698 vs 0.692 with overlapping
  intervals. Its **MRR** gain over plain L3 is real (intervals nearly
  disjoint), and its AP gain is clear, but the AUC difference is not
  established.

An earlier version of this document reported MRR differences among the top
configurations as findings — "MRR has doubled", R-GCN's 0.0170 versus SAGE's
0.0144. Those gaps are inside the noise band of a single measurement and should
not have been stated as results. They are removed rather than restated.

### What made the GNN work at all

**Edge masking.** At test time the target edge is absent from the graph. If
training supervises on edges that are also in the message-passing graph, the
model learns to read the adjacency it was handed rather than to predict.
Training edges are split into disjoint message-passing and supervision sets,
resampled every few epochs.

**Temporal validation.** A random validation slice reported AUC 0.912 while the
model scored 0.737 on the temporal test — early stopping was selecting against
a much easier distribution. Validating on the most recent training edges closed
that gap to 0.66 versus 0.71 and picks better models.

**Ranking loss.** Cross entropy optimises a global threshold; Hits@K and MRR
are per-positive rankings. Switching to BPR cost some AUC and doubled MRR. BCE
also proved much weaker overall here (validation 0.588 against BPR's 0.727).

**Hard negatives.** Training draws negatives matched on type and degree, the
same way the evaluation does. Training against easy negatives teaches a
boundary the evaluation never asks about.

### Why the ensemble beats both parts

The GNN and L3 agree less than they disagree — Spearman 0.33 on held-out pairs.
L3 counts concrete evidence paths in the observed graph; the GNN learns a
latent representation that generalises past the paths that happen to exist.
Averaging their percentiles beats both on AUC, AP and MRR, and every blend
weight from 0.25 to 0.75 beats both components, so the result does not hinge on
tuning. The weight is nevertheless selected on a temporal validation slice, not
on test.

The ensemble is also what makes the result *stable*: the GNN alone swings
±0.089 across seeds, the hybrid ±0.018. L3 acts as a floor the learned
component cannot fall through.

### Using the evidence the graph discards

Flattening to a simple undirected graph collapses 13194 CIVIC relations into
6645 pairs and throws away four things: 11 distinct predicates, subject/object
direction, curator confidence spanning 0.27–1.00, and 1731 negation flags
marking relations whose evidence says the association does **not** hold.

`weighted_l3` weights each hop of a path by the evidence behind that edge —
mean confidence times log support, penalised by the negated fraction. On its
own it lifts average precision from 0.205 to 0.238 and MRR from 0.0050 to
0.0170 over plain L3. Repeated assertion enters logarithmically because ten
papers asserting an association is not ten times the evidence, and negation
lowers a weight without erasing the edge — contested is not the same as absent.

Weights are built from **pre-cutoff evidence only**. Weighting an edge with
evidence published after the cutoff would feed the model the very knowledge the
holdout exists to withhold; there is a test pinning this.

`--relational` switches the encoder to R-GCN, learning one transform per
predicate. Its AUC is indistinguishable from the untyped encoder's, and the
MRR difference between them falls inside the bootstrap interval, so there is
currently **no measured reason to prefer either**. It is kept because modelling
opposite predicates as the same edge is wrong in principle, and because a
denser graph may make the difference measurable.

### Node text features

Every predictor before this used topology alone, which caps what is reachable:
14% of held-out pairs have no path between their endpoints at any length.
`NodeTextEncoder` embeds each node's display name and the GNN takes those
vectors as input alongside its learned embedding, node type and log-degree.

Names are static metadata, so this does not leak across the temporal split — a
disease was called "melanoma" before and after 2016.

**Measured effect, 8 seeds**, hybrid predictor:

| | AUC | range | AP | H@100 |
|---|---|---|---|---|
| topology only | 0.734 ± 0.006 | [0.724, 0.744] | 0.277 | 0.090 |
| **with text** | **0.754 ± 0.005** | **[0.745, 0.762]** | 0.299 | 0.105 |

The ranges are disjoint: +0.020 AUC, roughly four standard deviations. Four
seeds were not enough to see this — two 4-seed runs of the same configuration
disagreed by 0.023, more than the effect. Use `--seeds` generously.

**Which encoder, decided by measurement:** on name similarity alone,
PubMedBERT scores 0.580 [0.564, 0.595], MiniLM 0.533 [0.516, 0.550], and
BioBERT 0.514 [0.497, 0.530] against a 0.498 floor. PubMedBERT is the default.
BioBERT is barely distinguishable from chance despite also being a biomedical
model, so "biomedical" alone does not predict which encoder helps.

### The gain is not string matching

Some CIVIC therapies are named for their target — "BRAF Inhibitor" embeds at
cosine 0.65 against "BRAF" — so a model handed names could score certain pairs
from the strings alone. `text_only` bounds that: **AUC 0.581, against 0.750 for
the hybrid.** Text alone is far above the random floor but nowhere near
topology, and it is only in combination that it earns its keep. The features
help the model generalise across similar entities, not match substrings.

### Literature context features do not work

Node names carry no biology — "Imatinib" as a string says nothing about what it
treats — so the obvious next step was to characterise each entity by the
sentences it appears in. `litkg.phase2.literature_context` fetches pre-cutoff
PubMed abstracts per entity, extracts mentioning sentences, and feeds the pooled
embedding to the model in place of the name.

**It does not work, and the way it fails is the interesting part.**

Measured on the full 2016 test set, context looked like a large win over names:

| text feature | AUC | AP |
|---|---|---|
| names | 0.562 [0.549, 0.577] | 0.108 |
| context | **0.684 [0.668, 0.698]** | 0.193 |

Disjoint intervals, +0.12 AUC, nearly matching the structural baseline. That
number is an artefact.

Coverage is partial, and nodes without context fall back to their name. That
makes "has literature context" a feature in itself — and it is a popularity
proxy:

| | count | median degree |
|---|---|---|
| nodes with context | 697 | **6** |
| nodes without | 1914 | **2** |

A 3x degree difference, and the two groups' feature strings are 1677 versus 12
characters, so a model can separate them trivially. Well-studied entities have
more edges, and more edges means more held-out pairs.

Restricting the evaluation to pairs where **both** endpoints have context
removes the confound, since every node in that comparison is equally covered:

| predictor | covered-only subset (750 positives) |
|---|---|
| weighted_l3 | 0.563 [0.542, 0.583] |
| l3_paths | 0.547 [0.527, 0.565] |
| names | 0.544 [0.522, 0.564] |
| **context** | **0.485 [0.464, 0.506]** |

Context lands **below chance**. The entire apparent gain was the availability
signal, not the content. Every predictor scores lower on this subset because it
is restricted to well-connected nodes where degree-matched negatives are
genuinely comparable — which is the point of it.

In the full hybrid, context also *hurts*: 0.737 ± 0.024 against 0.747 ± 0.009
without text, with variance nearly tripled. A redundant, noisy popularity proxy
on top of a model that already has log-degree as a feature.

**Why it fails.** Mean-pooling a dozen sentences into one vector per entity
describes what an entity is generally discussed alongside, not how it relates to
any particular partner. An "average context" cannot encode that *this* drug
treats *that* disease. Entity-level context is the wrong granularity.

The plausible fix is pair-level: retrieve sentences mentioning **both**
endpoints and score the pair from those, which is a co-mention feature rather
than a node feature. That is a different design and is not implemented. Note it
would need the same date discipline, and a co-mention in a pre-cutoff abstract
is close to being the label itself — so it needs care, not just plumbing.

The fetching machinery is sound and tested and the cache is reusable, so trying
that costs the experiment, not the infrastructure.

### Cold start: coverage without much signal

The split excludes 366 pairs whose endpoints never appear in training, because
no topological method can score them. Text features can:

| predictor | AUC | pairs it can score |
|---|---|---|
| l3_paths | 0.418 | **0 of 366** |
| weighted_l3 | 0.418 | 0 of 366 |
| text_only | 0.531 [0.501, 0.562] | 366 of 366 |

L3's 0.418 is an artefact of every pair tying at zero. So text features are the
only thing that addresses cold start at all — but at 0.531 with a lower bound
of 0.501, barely better than guessing. This is a capability, not yet a result,
and it points at the limit of names: "Imatinib" as a string says nothing about
what it treats. That knowledge is in the literature, not the label. Embedding
entities by the abstract contexts they appear in is the obvious next step, and
the corpus is already processed.

### Read the per-type-pair table, not just the aggregate

The single headline number averages four problems of very different difficulty.
Plain L3, by entity-type pair:

| type pair | n | AUC | H@10 | MRR |
|---|---|---|---|---|
| MUTATION–PHENOTYPE | 132 | 0.802 | 0.061 | 0.0354 |
| DRUG–MUTATION | 403 | 0.722 | 0.012 | 0.0117 |
| DISEASE–MUTATION | 477 | 0.655 | 0.042 | 0.0171 |
| DISEASE–DRUG | 192 | 0.638 | 0.016 | 0.0092 |

A 0.164 spread. Mutation–phenotype is the easiest despite having the fewest
training edges, and disease–drug is the hardest — unfortunate, since drug
repurposing is the most clinically interesting of the four. Adamic-Adar scores
exactly 0.500 on mutation–phenotype: phenotypes attach only to mutations, so no
common neighbour can exist and the predictor has literally no information.

Every run prints this table. Judge changes on it, not on the aggregate, which
can move because one subproblem improved while another regressed.

### Why ranking metrics here are unreliable

Each positive is ranked against the **entire** negative pool — ~1200 positives
against ~12000 negatives. That makes MRR and Hits@10 extremely top-heavy:

- The top 20 positives contribute **78%** of MRR; the top 10 contribute 42%.
- Only ~26 positives out of 1204 land in the top 10 at all.
- The bootstrap CI for MRR spans [0.0066, 0.0135] — a width comparable to the
  value itself.

So MRR is effectively determined by a couple of dozen rows. Two configurations
whose MRR differs by a factor of two may be indistinguishable, and an apparent
trade-off between AUC and MRR across configurations is largely a stable
statistic being compared against a noisy one, not a property of the task.

Three consequences, all now implemented:

1. Every metric ships with a 95% bootstrap interval. Compare intervals, not
   point estimates.
2. `hits_at_100` is reported alongside Hits@10, which has no resolution at this
   pool size.
3. `indistinguishable_fraction` records how many positives the predictor cannot
   separate from the bulk — tied with more than half the negative pool. For
   shared-neighbour methods this is most of them, so their ranking metrics
   describe an undefined score rather than a wrong one.

### What it actually predicts

Every figure above scores held-out positives against roughly ten sampled
negatives each. That is not the task. A researcher asking "what should we look
at next" ranks every unobserved pair and reads the top of the list, and
sampled-negative AUC is an optimistic proxy for that.

```bash
python scripts/rank_predictions.py --cutoff 2016 --top 100 --seeds 5
```

This trains only on evidence published before the cutoff, ranks the real
candidate space, and asks how many of the top predictions CIVIC curated in the
years after. Prospective validation, using data already on disk.

**The candidate space and its ceiling.** For the four entity-type pairs the
held-out period contains, the full product is 988,604 pairs. 206,997 of them
have at least one three-path, which is all a structural score can rank; those
contain **889 of the 1388** later-curated pairs, so **64% is the ceiling** on
what this ranking can find. The other 36% are unreachable by construction.

**Results**, 5-seed consensus ranking, against a base rate of 0.429%:

| depth | precision | lift |
|---|---|---|
| 50 | 8.0% | 19x |
| 100 | **13.0%** | **30x** |
| 250 | 6.8% | 16x |
| 500 | 5.6% | 13x |

Thirty times better than picking at random from the candidate set. That is a
real signal, and it is the first evidence in this project that the system
surfaces associations before they are curated rather than merely scoring well
on a benchmark.

### Precision at the very top is not measurable here

**Read depth 50 and beyond.** Precision@10 could not be pinned down and the
reason is worth knowing.

Every model concentrates its top predictions on one or two dense clusters. The
top ten is invariably Von Hippel-Lindau disease against VHL variants -- VHL has
295 distinct molecular profiles, the densest neighbourhood in the graph -- or
ABL1 resistance mutations against imatinib and dasatinib. Whether those specific
pairs happen to appear in the post-cutoff curation is close to a coin flip.

Across five seeds on identical data, top-ten hits were **3, 2, 0, 8, 6**. Re-run,
the five-seed mean moved from 38% to 64%. A single number there would be
meaningless.

Two further cautions from building this:

- A **single seed produced an inverted precision curve** -- worse at depth 100
  than at depth 500 -- and an accompanying story about the model ranking
  obviousness over novelty, complete with a supporting statistic (median
  endpoint-degree product 189 for predictions against 49 for real discoveries).
  None of it survived five seeds. The curve is monotonic.
- **Correcting for node degree makes ranking worse** at every depth tried, so
  the hub concentration is not simply bias to be divided out. Degree carries
  real signal: well-studied genes genuinely have more true associations.

### Reading this honestly

AUC 0.750 against an 0.692 baseline is a real gain with disjoint confidence
intervals, bought with a large increase in complexity. **Hits@100 of 0.105
means about 10% of held-out associations reach the top 100 of ~12000** — this is a measurable improvement in a research
setting, not a system that surfaces useful hypotheses yet. The MRR gain
(0.0050 → 0.0072) is the more meaningful movement, and it is still small in
absolute terms.

Note also that runs are not bit-reproducible despite seeding: scatter-add
ordering in the graph convolutions varies. Single-seed numbers should not be
quoted; use `--seeds`.

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
