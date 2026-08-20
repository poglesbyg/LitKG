# Changelog

Notable changes to LitKG-Integrate. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Gene–gene edges from STRING** (`litkg.phase1.string_ppi`), and a
  `PathPowerPredictor` that can actually reach them. 1862 experiment-backed
  interactions among CIVIC's 973 genes. These are the first same-type edges in
  the graph, which was strictly multipartite: 0 of its edges joined two nodes of
  the same type. Neither CIVIC nor the GDC can supply them.

- **`PathPowerPredictor`**, degree-normalised length-k path counts via sparse
  matrix powers, because length 3 is structurally blind to the new edges. For a
  path `variant -> a -> b -> disease`, a gene–gene middle hop needs `b` to be a
  gene adjacent to the disease, and CIVIC has **no gene–disease edges at all**.
  Measured, not inferred: adding 1862 interactions changes the L3 path count for
  **0 of 1388** test pairs.

- **TCGA and CPTAC now come from the GDC open-access API** (`litkg.phase1.gdc_client`),
  replacing the placeholder that fabricated them. `TCGAProcessor.download_tcga_data`
  previously wrote three hardcoded rows to a CSV and read them back; the only
  trace of the Genomic Data Commons was a URL in `config.yaml` and a comment
  saying a real implementation would use the API. Those nine invented rows had
  been reaching the integrated graph as 11 nodes and 7 edges.

  The pipeline now pulls somatic mutations pinned to GDC data release 46.0
  (`LITKG_GDC_RELEASE` overrides), producing **528 gene–cancer type edges** over
  226 Cancer Gene Census genes and 35 cohorts: 488 from TCGA's 33 projects and
  40 from CPTAC's 2. CPTAC's *proteomics* remains unintegrated -- it lives in
  the Proteomic Data Commons, a different API -- so CPTAC contributes mutation
  data, not protein expression.

- **A length-aware background rate model for mutation counts**
  (`background_rate_enrichment`). Ranking genes by raw somatic mutation count
  measures coding length rather than biology: asked for the top mutated genes in
  breast cancer, the GDC returns OBSCN, PIK3C2B, NID1, NFASC and USH2A, none of
  them a breast cancer driver.

  Restricting to the Cancer Gene Census is necessary but **not sufficient**, and
  the control says so: among Census genes, the number of cohorts a gene reaches
  correlates with log CDS length at **+0.798**, *higher* than the +0.542 for raw
  occurrence counts, because thresholding on a count selects for length. Scoring
  each pair against what its length predicts takes the correlation to **-0.097**
  and leaves the known drivers on top -- IDH1 at 193x in lower-grade glioma,
  GNAQ and GNA11 in uveal melanoma, BRAF at 308x in thyroid carcinoma, KRAS at
  43x in lung adenocarcinoma.

- **A leakage guard for undated edges** (`litkg.evaluation.gdc_edges`). GDC
  edges carry no year, so `TemporalSplitter` routes them into the backbone:
  present at training time, never scored, which is precisely the position from
  which an edge can hand over an answer. Any GDC edge coinciding with a held-out
  CIVIC pair is dropped before the split is built, undirected, and the count is
  reported.

  It currently drops zero, for a structural reason worth recording: CIVIC has
  **no direct gene–disease edges at all** (0 of 13059 evidence relations and 0
  of 1738 variant relations), linking gene→variant→disease instead. That makes
  the GDC edges a new edge type rather than a duplicate one -- and makes the
  guard a standing check rather than a formality, since the release predates no
  cutoff: GDC 46.0 is from August 2026, and only cutoffs at or after ~2016 are
  defensible given when TCGA sequencing completed.

### Measured

- **Gene–gene edges are worth +0.017 AUC, at length 5 and not before.** Over 8
  seeds with degree-matched negatives:

  | cutoff | L3 without / with | L5 without / with |
  |---|---|---|
  | 2016 | 0.7045 / 0.7036 (overlap) | 0.7105 / **0.7273** (disjoint) |
  | 2020 | 0.7814 / 0.7835 (overlap) | 0.7846 / **0.7921** (disjoint) |

  Average precision goes 0.228 -> 0.299 and Hits@100 0.042 -> 0.093, which is
  the ranking head this project has been weakest at. The largest per-type gain
  is DRUG-MUTATION, 0.707 -> 0.772. First addition since the node-text features
  to clear the project's own bar: 8 seeds, two cutoffs, disjoint at both.

- **STRING's `combined_score` would have made this circular, by a wide margin.**
  It fuses seven channels and one is co-occurrence in PubMed abstracts -- the
  same papers CIVIC curators read to write the labels. Among CIVIC's genes,
  textmining alone contributes **14380 edges** against 1862 from physical
  experiments and 3795 from curated databases. KRAS-BRCA1 scores 0.721 combined,
  of which 0.721 is textmining and 0.000 is experiments; TP53-BRAF scores 0.887
  with 0.883 from textmining. `StringPPI.edges` defaults to `experimental` and
  **raises** on `textmining` or `database` unless `allow_literature_channels=True`.

- **Adding edges to the backbone is not free, and three sources now show it.**
  GDC gene-cohort edges, GDC variant-cohort edges and STRING interactions are
  each individually indistinguishable from baseline under L3, and GDC + STRING
  together are *disjointly worse* (-0.0068 at 2016, -0.0073 at 2020). Under L3
  they add paths for positives and negatives alike; the predictor, not the data,
  was the limit.

- **The variant-level CIVIC/GDC join does not work either, and the arithmetic
  says why.** The gene-level join failed on granularity: TP53 links to 14 of 15
  diseases in the joined graph, so the edge approximates a prior. CIVIC's unit
  is the variant, and at that level the signal is sharp -- BRAF is mutated
  across most cohorts while BRAF V600E concentrates in thyroid (283) and
  melanoma (200). It also targets the largest test category, since 39% of
  held-out pairs are DISEASE-MUTATION.

  It is still unmeasurable. Of 546 DISEASE-MUTATION test pairs, 186 of 463
  variants are simple protein changes, 85 of those were observed in TCGA, and
  only 14 of 78 test diseases correspond to a TCGA cohort. **20 pairs survive
  both filters -- 1.4% of the test set.** The predictor scores AUC 0.501 at 0.9%
  coverage, which is an abstention rate rather than a result.

  Improving the cohort mapping from 22 to 30 of 33 changed the pair count **not
  at all**, which is the load-bearing observation: the bottleneck is not join
  quality. TCGA is 33 common adult solid tumours; CIVIC spans 246 diseases
  including rare and haematological ones TCGA never sequenced. The two sources
  cover different diseases, and no amount of ontology work changes that.

- **`litkg.phase1.disease_ontology`**, added while chasing the above. Resolves
  cohort names to DOIDs against the Disease Ontology's synonym list and `is_a`
  hierarchy, so the CIVIC join runs on identifiers rather than strings: 22 of 33
  cohorts against 14 by name. Ancestry matches are flagged `via_ancestor`,
  because mapping a lung-adenocarcinoma cohort onto a lung-cancer node is a
  generalisation and should be visible as one. Useful independently of the
  result it failed to rescue.

- **The GDC edges do not improve link prediction.** Added to the training
  backbone across 8 seeds, `weighted_l3` moves from 0.693 ± 0.001 to
  0.691 ± 0.001 at a 2016 cutoff (seed ranges overlap -- indistinguishable) and
  from 0.769 ± 0.002 to 0.765 ± 0.001 at 2020 (disjoint seed ranges -- a small
  real **degradation**). `--with-gdc` is off by default in
  `scripts/evaluate_link_prediction.py`, and no claim is made that this data
  helps prediction. It is in the graph, where retrieval and the discovery
  pipeline can use it.

- **Only 14 of 33 TCGA cohorts join CIVIC by exact name**, so 153 of 488 TCGA
  edges reach the evaluation graph. Matching is exact after normalisation
  deliberately: a fuzzy match would merge "Lung Adenocarcinoma" with "Lung
  Squamous Cell Carcinoma", different diseases with different drivers, and no
  downstream metric would reveal it. Closing the gap needs a real
  cohort-to-Disease-Ontology mapping rather than a hand-written alias list.

### Fixed

- **The `HybridGNNModel` representation collapse, root-caused and corrected.**
  It scored at chance (0.492) because every node ended up the same vector.
  Over-smoothing was the obvious diagnosis and was wrong: a single message
  passing layer collapsed just as completely, a learned per-node embedding did
  not help, and random input features produced no collapse at all.

  The cause was the input. Mean-pooled PubMedBERT vectors are anisotropic --
  distinct entity strings sit at a mean pairwise cosine of **0.930** before the
  model sees them -- so every node began nearly parallel and message passing
  only closed the last gap. Centring the feature matrix takes the input to
  0.214 and the score from **0.492 to 0.633 ± 0.024** across 5 seeds. One-hot
  type indicators share a direction for the same reason, so the whole vector is
  centred rather than the text block alone.

- **`RelationPredictor` returned only a post-sigmoid probability**, which no
  ranking loss can use: recovering the logit saturates and the gradient
  vanishes. It now returns `link_logits` alongside `link_probs`.

  That fixed a real defect and did not change the score, which is the
  informative part. Scored through the shipped decoder the model sits at
  **0.477 ± 0.018** against **0.633** for an inner product on identical
  representations -- concatenating two endpoints cannot express a pairwise
  interaction the way an element-wise product can. The inner product is the
  default and `use_model_decoder=True` reproduces the comparison.

  Cross-modal attention still costs 0.12 AUC against a simpler model with none.
  The difference is now an architectural question rather than a bug.

### Documentation

- **Two README status rows were false.** `HybridGNNModel` and Phase 3 were still
  marked "synthetic demo only" after both had been wired to real data and
  measured. The edits that should have corrected them were string replacements
  whose anchors no longer matched, and a silent no-op leaves the old claim
  standing -- which is how a status table ends up asserting the opposite of what
  shipped.

  The table now lists every component, all of which run on real data, with the
  harness that produced each number. It also states plainly that
  `HybridGNNModel` scores at chance and that the prospective discovery result
  did not replicate, since those are the two entries a reader would otherwise
  assume work.

- **`make run-phase2`, `run-phase3`, `run-langchain` and `run-discovery` are now
  labelled as synthetic demonstrations** in the README, alongside the commands
  that actually touch the real graph. They were previously listed under
  "Pipelines" as if they were the working entry points.

- **ARCHITECTURE.md** now shows the real prediction path. Its data flow diagram
  routed prediction through the hybrid GNN, which is the component measured at
  chance; the measured numbers come from `litkg.phase2.link_prediction`. It also
  documents `DiscoveryPipeline` as the second place the two halves meet, and
  records the fusion defect and the representation collapse.

- Test count corrected to 533 and the Phase 1 figures to 3822 nodes, 14810
  edges, 150 cross-modal links.

- Added a note on how to read any figure in the repository: five results have
  failed replication, so a single-seed or single-cutoff number is a hypothesis.

### Added

- **An end-to-end pipeline** (`make discover`, `scripts/discover.py`,
  `litkg.pipeline`). The project had two chains that never met: one read CIVIC,
  trained a link predictor and ranked unobserved pairs; the other read the
  Phase 1 literature and answered questions with citations. Nothing joined a
  prediction to its evidence.

  Joining them needed more than wiring. The bundled corpus contains **no
  literal mention** of the entities the top predictions involve -- it is general
  cancer genomics while the predictions concern VHL variants and ABL1
  resistance mutations -- so retrieval against it returns unrelated passages.
  Evidence is now fetched *for each candidate*, querying for both entities
  together and falling back to single-entity queries that are labelled as such,
  because a passage mentioning one half of a pair is background rather than
  support.

  `--cutoff` restricts the graph and the literature to before a given year and
  marks candidates curated afterwards, so a run can be checked instead of
  trusted.

  The report states plainly that the ranking's precision does not replicate
  across cutoffs, and the model is asked what the evidence does *not* support.
  On a variant CIVIC later confirmed, it correctly reported that the retrieved
  passages spoke only of VHL mutations in general rather than that variant --
  right about the evidence, which is the question it was asked.

### Changed

- **The prospective validation result does not replicate, and the claim is
  withdrawn.** It was measured once, at a 2016 cutoff, and reported as evidence
  that the system surfaces associations before they are curated. Re-run at
  other cutoffs (`scripts/replicate_prospective.py`):

  | cutoff | base rate | P@100 | lift@100 | lift@500 |
  |---|---|---|---|---|
  | 2016 | 0.429% | 15.0% | 35x | 21x |
  | 2018 | 0.188% | 1.0% | 5x | 2x |
  | 2020 | 0.125% | 0.0% | 0x | 6x |

  2016 is an outlier at both depths. "Later cutoffs leave less future in the
  data" does not explain it: lift is normalised by base rate, and 2018 is worse
  than 2020 at depth 500.

- **The Phase 3 type-pair rates are likewise cutoff-specific.** Disease-mutation
  read 45% at 2016 and 4-6% at 2018 and 2020; mutation-phenotype looked like a
  category that never pays off and is ordinary once the cutoff moves. The
  six-fold filter suggested by those numbers is withdrawn.

  This is the fifth result in this project to fail replication, after "the graph
  is too sparse", a doubled MRR, an inverted precision curve, and a
  hub-dominance story. All five looked solid at a single configuration.

### Fixed

- **`HybridGNNModel` could not express a per-pair prediction.** Fusion combined
  the literature and knowledge graphs at the *graph* level, so
  `fused_representation` had exactly one row and `entity_pairs` could only ever
  index row 0 -- the synthetic demo passed `[[0, 0]]` with the comment "only use
  valid indices", and that comment was load-bearing. Cross-attention already
  preserved the node dimension, so fusion now operates on node embeddings and
  indexes the enhanced KG nodes. The graph-level vector remains available as
  `graph_fused_representation`.

### Added

- **`HybridGNNLinkPredictor`**, which trains `HybridGNNModel` on the real graph
  and scores it through the standard harness -- same split, negatives, metrics
  and loss as every other model, so the comparison is like for like.

  **It scores at chance.** AUC 0.492 +/- 0.020 across three seeds, against
  0.744 +/- 0.006 for the far simpler GNN + weighted-L3 + text hybrid. Chance
  is 0.5.

  The cause is degenerate representations: mean pairwise cosine between
  distinct nodes runs 0.793 at the input, 0.998 after the KG encoder and 1.000
  after fusion, so every node ends up the same vector. Adding a learned
  per-node embedding -- which the simpler baseline has, so withholding it would
  not have been a fair test -- did not help, nor did reducing message passing to
  a single layer, so this is not ordinary depth-driven over-smoothing. The
  collapse is located but not fully explained, and finding the exact cause is a
  separate investigation.

  This closes the last synthetic-only component. Every phase now runs on real
  data, and this one is measured as not working.

### Added

- **Phase 3 runs on real predictions and its scores are checked, not just
  displayed** (`scripts/assess_predictions.py`). Confidence scoring, plausibility
  and novelty had only ever seen six hardcoded relationships and random tensors,
  because nothing produced real predictions to assess.

  Evidence comes from pre-cutoff CIVIC rows only; later evidence describes the
  associations being predicted. Because the predictions come from a temporal
  holdout, every one has a known outcome, so the scores can be measured.

  **The neural confidence scorer carries little signal** -- AUC 0.613, with
  group means of 0.514 and 0.512. The assessors are untrained and emit
  near-constant output.

  **The plausibility score is a type prior rather than biology.** It takes four
  distinct values across 500 predictions, one per entity-type pair, so its AUC
  of 0.863 reflects differing curation rates by type and not reasoning about
  mechanism. Those rule values were also hand-set while fixing the gap below,
  so the number is not validation.

  **The finding worth acting on needs no model at all:** curation rates are
  32.7% for disease-mutation pairs, 2.4% for drug-mutation and 0% for
  mutation-phenotype. Reading only the first lifts precision from 5.0% to 32.7%
  on this sample, and mutation-phenotype pairs -- never curated once -- are 29%
  of the ranked output.

  Calibration, fitted on one half and tested on the other, is overconfident by
  about four-fold (0.080 predicted against 0.020 observed).

### Fixed

- **The plausibility rule table had no MUTATION entry**, and mutations are an
  endpoint of most predictions this graph produces, so every such pair fell to
  the 0.3 default and the score was constant across almost the whole candidate
  set. Added MUTATION, FUSION and PHENOTYPE pairs, mirroring the gene rows.
- The assessment script read `plausibility_score` from a method that returns
  `score`, silently yielding 0.0 for every prediction.

### Added

- **Prospective validation: rank the real candidate space and check the top
  predictions against what CIVIC curated afterwards**
  (`scripts/rank_predictions.py`). Every link-prediction figure so far scored
  held-out positives against ~10 sampled negatives each, which is not the task
  a user performs; ranking a million unobserved pairs is.

  Of 988,604 candidate pairs across the four held-out entity-type pairs,
  206,997 have a three-path and contain 889 of the 1388 later-curated pairs --
  a **64% ceiling** on what structural ranking can reach.

  Against a base rate of 0.429%, the 5-seed consensus ranking is **13%
  precision at depth 100, a 30x lift**, decaying to 5.6% at depth 500. This is
  the first evidence that the system surfaces associations before they are
  curated rather than only scoring well on a benchmark.

  **Precision@10 is not measurable and the script says so.** Every model
  concentrates its top predictions on one or two dense clusters (Von
  Hippel-Lindau with its 295 profiles, or ABL1 resistance mutations against
  imatinib), so whether those specific pairs were later curated is close to a
  coin flip: five seeds on identical data scored 3, 2, 0, 8 and 6 hits out of
  ten, and the five-seed mean moved from 38% to 64% on a re-run.

  Two findings that did not survive scrutiny are recorded rather than dropped:
  a single seed produced an inverted precision curve plus a supporting
  statistic about the model ranking obviousness over novelty, and neither
  survived five seeds; and correcting for node degree makes ranking worse at
  every depth, so hub concentration is not bias to divide out.

### Fixed

- **Documented the wrong cause for the single-relationship recall drop.**
  Enabling expansion lowers recall at fixed budget from 0.826 to 0.712, which
  looks impossible since expansion only adds candidates. The shipped
  explanation blamed P@k dividing by k; that is true of P@k and irrelevant to
  recall.

  The real mechanism: `k` counts *seed* passages, not results, so with expansion
  the retriever returns `k + expansion_limit` documents. A consumer truncating
  back to `k` chooses which source to cut, and RRF ties expanded passage n with
  seed n, so five expanded candidates displace exactly the last five seeds.
  Measured at k=10: all 57 queries lose five seed papers from the evaluated
  window, 282 in total.

  **No evidence is lost on the answer path** -- `answer()` consumes everything
  the retriever returns without truncating, so fusion changes prompt order and
  nothing else. Also recorded: the trade-off is a step rather than a slope, as
  any seed weight above 1.2x restores 0.826 exactly and returns bridge hit-rate
  to 0.236.

  Three regression tests pin the behaviour, including that `retrieve()` must
  not truncate, since the documentation now depends on it.

### Changed

- **Seeds and graph-expanded passages are now combined by reciprocal rank
  fusion** rather than concatenated. Concatenation gave dense retrieval half the
  returned slots whether or not it had found anything, and on bridge queries it
  usually had not.

  Predicting which questions are entity-anchored and reallocating was the
  obvious alternative; it was measured and rejected. Seed coverage of the
  query's own entities separates the two query sets far too weakly to switch on
  -- zero coverage fires on 6 of 55 bridge queries -- and a classifier keyed on
  phrasing would fit the templates the queries are generated from. Fusion needs
  no prediction: a passage ranked highly by either route earns a place. The
  fusion constant stays at the published default, since tuning it on 55 queries
  would fit the query set.

  Bridge queries: hit-rate **0.236 -> 0.327**, P@10 0.031 -> 0.049, MRR
  0.047 -> 0.095.

  The cost, stated rather than buried: with expansion switched on, the
  single-relationship set drops from P@10 0.370 / R@10 0.826 to 0.304 / 0.712.
  `max_hops` defaults to 0, so that path is opt-in, and enabling expansion is
  now a measured trade -- reach on questions whose answers are not lexically
  present, against precision on questions where dense retrieval was already
  right.

### Fixed

- **Derived evaluation data was being committed** -- two FAISS indexes and two
  fetched corpora, 2.5MB of rebuildable files. The `data/processed` patterns
  match only that directory's own files, so anything in a new subdirectory
  slipped past them, which had already happened once with a 14MB embedding
  cache and a 10MB abstract cache.

  Pattern-per-filetype cannot win that race, so the default is inverted:
  everything under `data/processed` is ignored and the exceptions are
  allow-listed, covering a future cache directory before it exists.

  The judged query sets stay tracked -- they are the benchmark, and
  regenerating them would silently change what a reported number means.
  `--rebuild-corpus` refetches the papers an existing query set names, which is
  what a fresh clone needs, and the evaluation script says so when the corpus
  is absent.

### Added

- **A multi-hop query set, and two fixes that make graph expansion contribute.**
  The single-relationship set could not judge multi-hop: it marks relevant only
  what CIVIC cited for the relationship the query names, which is what dense
  retrieval already finds. `--multihop` builds bridge queries instead -- each
  names one molecular profile, and the relevant papers are those cited for
  *different* profiles in the same disease, reachable only along
  profile -> disease -> other profile. Any candidate naming the query's own gene
  is dropped, because dense retrieval could find it directly; 81 papers were
  removed that way, leaving 55 queries over 506 documents.

  Tracing the chain showed the graph holds a path to the answer for **54 of 55**
  queries while only 9 were retrieved. Two causes:

  - Expansion seeded only from retrieved passages, so it was a hostage to dense
    retrieval, which surfaced a usable seed for just 16 of 55. Entities named in
    the question now seed the walk directly.
  - Candidates were truncated in walk order, which is close to random sampling
    when a hop reaches hundreds of passages. A pool of 200 contains the answer
    for 85.5% of queries against 21.8% at 5.

  **Ranking by similarity to the query is self-defeating and was measured as
  such** -- it nullified expansion entirely, because a passage reachable only
  through the graph is by definition one that does not resemble the question.
  Inverse node degree is also wrong here: the bridge between two variants is the
  disease both are evidenced in, exactly the high-degree node that weighting
  penalises. Ranking by shared graph context works: 52.7% at top 5 standalone
  against 21.8% for walk order.

  Net effect on the bridge set: hit-rate 0.200 -> 0.236, P@10 0.025 -> 0.031.
  **An improvement, not a solution** -- a quarter of these queries surface a
  relevant bridge paper against 98% for single-relationship questions, and the
  gap between 85% pool recall and 24% top-10 is budget spent on seeds that are
  usually wrong. Single-hop retrieval is unchanged (P@5 0.547, MRR 0.803).

### Fixed

- Chunk ids were composed as `{pmid}:{pmid}:{position}`, because the pipeline
  included the pmid that `ChunkGraphIndex` already prefixes. Harmless to the
  index, but it made the key unjoinable by anything reconstructing it.

### Removed

- **Two encoders `BiomedicalNLP` downloaded on every run and never read.**
  `_load_models` populated `self.pubmedbert_model`/`self.pubmedbert_tokenizer`
  and `self.biobert_model`/`self.biobert_tokenizer` from the `pubmedbert` and
  `biobert` config keys. Nothing in `src/`, `scripts/` or `tests/` referenced
  those attributes -- roughly 800MB fetched and two model loads paid per run to
  populate names no code path touched, plus two config keys that advertised a
  setting which changed nothing.

  Both loads and both config keys are gone, along with the now-unused
  `AutoModel`/`AutoTokenizer`/`BertModel`/`BertTokenizer` imports. The encoders
  that are actually used are unaffected and pick their own checkpoints:
  `litkg.phase2.node_features` (PubMedBERT, for text features) and
  `litkg.models.huggingface_models.ModelRegistry` (which registers both under
  its own keys). `scripts/setup_models.py` still pre-fetches them for those
  paths.

### Fixed

- **The BERT NER path extracted nothing on every document.** The pipeline was
  built from `dmis-lab/biobert-base-cased-v1.1`, a base language model with no
  token-classification head, so transformers initialized one at random
  ("Some weights ... newly initialized: ['classifier.bias', 'classifier.weight']")
  and the pipeline returned `LABEL_0`/`LABEL_1` spans at ~0.5 confidence.
  `_map_bert_label` passed those through unchanged, none is in `entity_types`,
  and every one was dropped -- a stage that loaded a model per run, appeared
  active, and contributed zero entities.

  The path now runs `alvaroalon2/biobert_genetic_ner` (BioBERT fine-tuned on
  JNLPBA/BC2GM, configurable as `phase1.literature.models.biomedical_ner`), and
  `BiomedicalNLP` refuses any checkpoint whose `id2label` is the untrained
  `LABEL_N` default instead of running it for nothing.

  On 150 abstracts sampled from `data/processed/literature_context`, it adds
  **247 entities over 182 distinct surface forms** that neither scispacy model
  nor the gene rules produce -- mixed-case symbols (`Gsα`, `SetD5`, `apoE4`,
  `cullin-2`, `p16`), mouse allele notation (`Fgfr2(+/S252W)`) and modified
  residues (`H3K27me3`) -- against 4906 entities from those paths combined.
  Spans are dropped when they cut a word in half (5.7% of raw spans, an
  artifact of subword aggregation: "isplatin", "arubicin"), when the surface is
  a known non-gene acronym (`CAR`, `MDS`), or when they overlap a span scispacy
  already claimed, so one mention cannot become two entities.

### Added

- **A relevance-judged retrieval query set, derived from CIVIC citations**
  (`make build-queryset`, `make eval-retrieval`, `litkg.evaluation.retrieval`).
  Retrieval was unmeasured, so `k`, `max_hops` and the hub-degree cap were set
  by judgement with nothing to check them against.

  There are no human labels for this corpus, and an LLM judging retrieval that
  feeds the same LLM is close to circular. CIVIC supplies judgements instead:
  each evidence row cites a paper *and* states the relationship it supports, so
  cited papers are relevant to a question about that relationship on a
  curator's judgement. Grouping by (profile, disease, evidence type) with at
  least three cited papers gives **57 queries over 228 papers**.

  **Vector retrieval works**: MRR 0.81, hit-rate 0.98, R@10 0.815
  [0.753, 0.875].

  **Graph expansion does not.** Every hops setting scores identically and the
  hub cap changes nothing. By origin, hop-0 passages are 55.4% relevant against
  4.6% for hop-1 -- twelve times less precise -- and this is not a plumbing
  failure, since all 235 chunks link to the graph.

### Changed

- `PipelineConfig.max_hops` now defaults to **0**. Expansion on by default
  dilutes the evidence handed to the model, and the measurement does not
  support it. It stays available via `--hops`, because the judgements are
  biased against expansion by construction: they mark relevant only what CIVIC
  cited for one relationship, which is exactly not the vocabulary-crossing
  evidence multi-hop exists to reach. So this shows expansion does not help on
  questions of this shape, not that it is useless -- establishing that needs
  judgements built for multi-hop questions, which do not exist yet.

### Added

- **`RAGPipeline`: the wiring between Phase 1 output and the retrieval stack**
  (`make rag Q="..."`, `scripts/run_rag.py`, `litkg.langchain_integration`).
  The retrievers, chunk-to-graph index and agents were unit tested but nothing
  connected them to real data -- `make run-langchain` built its own hardcoded
  documents and a bare FAISS index, so the graph-aware path was never exercised
  outside tests.

  On the bundled corpus: 80 documents, 81 chunks, 78 of 81 (96%) linked to the
  graph, reaching 158 of 2971 nodes. Answers are grounded and cited -- asked why
  BRCA1 tumours are sensitive to olaparib, the system explains synthetic
  lethality and quotes TBCRC 048 and OlympiA figures from the passages it cites.

- **Hub-traversal cap on graph expansion.** `CIVIC:DISEASE:DOID:162` ("cancer")
  has degree 429 and is linked from 29 of 81 chunks; expanding through it
  reaches 207 nodes in one hop and 824 in two, so every oncology passage became
  a neighbour of every other. `ChunkGraphIndex.neighbors` now refuses to walk
  *through* nodes above `DEFAULT_MAX_TRAVERSAL_DEGREE` (50), while still
  allowing them to be reached and reported as evidence.

  Retrieval relevance remains **unmeasured**: there is no relevance-judged query
  set, and on the one question examined only about one graph-expanded passage in
  five is relevant even after the cap.

### Fixed

- The CLI rendered every source as "hop 0, pmid ?" because `answer()` flattens
  document metadata into each source dict rather than nesting it under
  "metadata". Pinned with a test on the payload shape.

### Added

- **Literature-context features, measured and rejected.** Entity names carry no
  biology, so `litkg.phase2.literature_context` characterises each entity by
  pre-cutoff PubMed sentences instead. On the full test set this looked like a
  large win -- AUC 0.684 [0.668, 0.698] against 0.562 for names, disjoint
  intervals.

  **The gain was a confound.** Coverage is partial and uncovered nodes fall back
  to their name, so "has literature context" becomes a feature -- and covered
  nodes have median degree 6 against 2 for uncovered ones. Restricting the
  comparison to pairs where both endpoints have context removes it, and context
  then scores **0.485 [0.464, 0.506]**: below chance. In the hybrid it also
  hurts, 0.737 +/- 0.024 against 0.747 +/- 0.009, with tripled variance.

  Mean-pooling sentences per entity describes what an entity is discussed
  alongside, not how it relates to a specific partner. Entity-level context is
  the wrong granularity; a pair-level co-mention feature is the plausible fix
  and is not implemented. The fetching machinery is kept -- it is sound, tested,
  date-guarded and cached, so the next attempt costs an experiment rather than
  the infrastructure.

### Fixed

- **A unit test overwrote the real literature-context cache.**
  `ContextConfig(cache_dir=None)` meant "the default real location" rather than
  "no cache", so a test that set fixture data and called `save()` destroyed a
  2911-entity cache. `use_cache` is now a separate flag, `save()` is a no-op
  when caching is off, and a test asserts a full suite run leaves the cache
  directory untouched.

  This is the same overloading fixed in `FeatureConfig` earlier the same day.
  Fixing one and not checking its sibling is how it happened twice.

- `LiteratureContextFetcher.gather` returned a renamed attribute, so a completed
  fetch saved all its work and then raised `AttributeError` on the way out.

### Documentation

- **README status table now distinguishes "run on real data" from "synthetic
  demo".** Several `make run-*` targets build `torch.randn` tensors and
  hardcoded documents; they exercise a code path without saying anything about
  whether it works on data. Verified by running every phase:

  - Phase 1, chunking, KG preprocessing, entity linking, link prediction: real.
  - `HybridGNNModel` (Phase 2, 1.8M params, cross-modal attention): synthetic
    only. It has never been trained on `phase2_graph_data.json`; the measured
    AUC 0.748 comes from the simpler model in `phase2/link_prediction.py`.
  - Phase 3 discovery: synthetic only, 6 hardcoded relationships.
  - RAG and agents: the library imports, constructs, links chunks to graph
    nodes and answers -- verified directly -- but no script or CLI command runs
    it, and `make run-langchain` uses hardcoded documents with raw FAISS.

- Stale figures corrected: 276 -> 452 tests, and node/edge/link counts updated
  to the CIVIC 01-Aug-2026 release.

- Known limits now record that BERT NER fails on abstracts over ~512 tokens
  (caught and logged, so the pipeline succeeds while that extractor
  contributes nothing), and that ranking metrics are dominated by a few dozen
  rows.

### Changed

- **CIVIC data updated from the 01-Feb-2024 release to 01-Aug-2026**, and the
  release is now configurable rather than hard-coded into three URLs. The
  default stays *pinned* to a dated release: a nightly build changes underneath
  you, so a regression could not be told apart from a data update. Override
  with `LITKG_CIVIC_RELEASE=nightly` or another release date. The active release
  is recorded in `data/external/civic/RELEASE`.

  Evidence grows 4254 -> 4878 rows, variants 1694 -> 1992, and distinct dated
  pairs 6643 -> 6981. Citations now run to 2025, which makes later cutoffs
  viable: the 2020 holdout has 513 test pairs against 116 before.

  Releases from 2024 onward ship a *features* file rather than a genes file --
  617 genes alongside 345 fusions, 8 factors and 3 regions. These were being
  typed as genes, which would have put fusions into the vocabulary that
  literature gene mentions resolve against; `feature_type` is now honoured.

  Results at the comparable 2016 cutoff are unchanged (hybrid 0.748 +/- 0.009
  against 0.750 on the old release), which is the reassuring outcome for a data
  refresh. At the 2020 cutoff the hybrid reaches 0.791 +/- 0.008 with Hits@100
  of 0.131 -- **higher because it is an easier problem**, training on 26% more
  pairs against a smaller test set, not because the method improved.

### Added

- Schema verification on download. An earlier version of this code read
  `drugs`, `variant_id` and `clinical_significance` from an evidence file that
  had none of them and produced 4125 dangling edges in silence. A missing
  required column is now an error at download time, not an empty string at
  processing time.

### Added

- **Node text features** (`litkg.phase2.node_features`). Every predictor until
  now used topology alone, which cannot reach the 14% of held-out pairs whose
  endpoints have no path between them. Node display names are embedded and fed
  to the GNN alongside its learned embedding, node type and log-degree. Names
  are static metadata, so this does not leak across the temporal split.

  Measured over **8 seeds** on the hybrid: AUC 0.734 +/- 0.006 [0.724, 0.744]
  without text, 0.754 +/- 0.005 [0.745, 0.762] with it -- disjoint ranges, about
  four standard deviations. AP 0.277 -> 0.299, Hits@100 0.090 -> 0.105. Four
  seeds were **not** enough to establish this: two 4-seed runs of the same
  configuration disagreed by more than the effect size.

  The encoder was chosen by measurement, not reputation. On name similarity
  alone: PubMedBERT 0.580 [0.564, 0.595], MiniLM 0.533 [0.516, 0.550], BioBERT
  0.514 [0.497, 0.530] against a 0.498 floor. PubMedBERT is the default;
  BioBERT is barely above chance despite also being biomedical.

- **`FeatureOnlyPredictor`**, a deliberate control. Some CIVIC therapies are
  named for their target ("BRAF Inhibitor" embeds at 0.65 against "BRAF"), so
  text features could in principle win by string matching. Text alone scores
  0.581 against the hybrid's 0.750, which bounds that effect: the gain comes
  from combining text with topology, not from substring overlap.

  It is also the only predictor that can score cold-start pairs at all -- L3
  scores 0 of 366 -- though at AUC 0.531 [0.501, 0.562] that is coverage
  without much signal. Entity *names* carry little biology; "Imatinib" says
  nothing about what it treats. Embedding entities by their literature contexts
  is the natural next step.

### Fixed

- `FeatureConfig(cache_dir=None)` meant both "use the default cache" and "no
  cache", so an encoder configured for no caching read vectors written by a
  different model and failed on a shape mismatch. Caching is now controlled by
  an explicit `use_cache` flag, and a width mismatch discards the stale cache
  with a warning rather than raising from inside numpy.

### Fixed

- **Ranking metrics were reported without uncertainty, and some differences
  claimed from them were noise.** Each positive is ranked against the whole
  negative pool (~1200 against ~12000), which makes MRR extremely top-heavy:
  the top 20 positives supply 78% of it, only ~26 of 1204 reach the top 10, and
  its bootstrap CI spans [0.0066, 0.0135] -- about as wide as the value. Every
  metric now carries a 95% bootstrap interval, `hits_at_100` is reported
  because Hits@10 has no resolution at this pool size, and
  `indistinguishable_fraction` records how many positives a predictor cannot
  separate from the bulk.

  Two previously stated results do not survive this and have been withdrawn
  from the docs: that recovering edge evidence "doubled MRR", and that the
  R-GCN ranked better than the untyped encoder. Both gaps sit inside the
  interval. The AUC and AP results do survive -- the hybrid's AUC interval is
  disjoint from every structural baseline's.

  The apparent AUC/MRR trade-off across configurations was also an artefact of
  comparing a stable statistic against a noisy one, not a property of the task.

### Added

- **Evidence-weighted edges.** Flattening the graph discarded 11 predicates,
  direction, confidence and 1731 negation flags, collapsing 13194 relations
  into 6645 pairs. `RelationRecord` and `EdgeEvidence` carry those through the
  temporal split, and `WeightedL3PathPredictor` weights each path hop by mean
  confidence x log support, penalised by the negated fraction. On its own that
  lifts average precision 0.204 -> 0.231 and MRR 0.0044 -> 0.0097. The AUC
  difference over plain L3 (0.698 vs 0.692) is inside the confidence interval
  and is not claimed.

  Weights are aggregated from **pre-cutoff evidence only**; weighting an edge
  with later evidence would leak the knowledge the holdout withholds.

- **Relation-aware encoder** (`--relational`). An R-GCN learning one transform
  per predicate, since the untyped graph treats SENSITIZES_TO and RESISTANT_TO
  -- opposite claims about the same pair -- as identical edges. Its metrics are
  currently indistinguishable from the untyped encoder's once confidence
  intervals are accounted for; it is kept because collapsing opposite
  predicates is wrong in principle, not because it measures better.

- **Per-entity-type-pair reporting.** Every evaluation now breaks results down
  by type pair, because the aggregate averages four problems whose AUC spans
  0.638 (disease-drug) to 0.802 (mutation-phenotype).

### Changed

- The hybrid now ensembles the GNN with *weighted* L3 and passes relation types
  through. AUC 0.743 [95% CI 0.729-0.755] against L3's 0.692 [0.677-0.707] --
  disjoint intervals -- with AP 0.262 and Hits@100 0.069, and 5/5 seeds over
  the bar. Seed variance halved, from +/-0.018 to +/-0.010.

### Added

- **Trained link predictors** (`litkg.phase2.link_prediction`,
  `scripts/train_link_prediction.py`). A 2-layer GraphSAGE encoder with an MLP
  edge decoder, and a hybrid that ensembles it with the L3 path baseline.

  The hybrid reaches **AUC 0.729 ± 0.018 against the 0.692 baseline, beating it
  in 5 of 5 seeds**, with average precision 0.244 vs 0.205 and MRR 0.0072 vs
  0.0050. The GNN alone does not clear the bar reliably: 0.670 ± 0.089, with
  one seed collapsing to 0.512. The two components correlate at only Spearman
  0.33, which is why ensembling them helps and why the ensemble is far more
  stable than the learned part alone.

  Four things were required to make the GNN train meaningfully: disjoint
  message-passing and supervision edges (otherwise it reads the adjacency it
  was handed), temporal rather than random validation (a random slice reported
  0.912 against a true 0.737), a BPR ranking loss rather than cross entropy,
  and type- and degree-matched training negatives matching the evaluation.

  Hits@10 remains 0.014 — this is a measurable research result, not a system
  that surfaces useful hypotheses.

### Fixed

- **The first hybrid scored AUC 0.000.** It rank-transformed within each call,
  and the harness scores positives and negatives separately, so with ten times
  as many negatives every negative outranked every positive. Percentiles are
  now taken against a reference distribution fixed at fit time, making scores
  independent of the batch they were computed in.

### Added

- **`L3PathPredictor` baseline**, and it changes the project's headline result.
  The CIVIC graph is strictly multipartite — 0 of 6769 edges join same-type
  nodes — and 100% of held-out pairs are cross-type, so shared-neighbour
  predictors are undefined on it rather than weak. Counting degree-normalised
  length-3 paths on the same split with the same negatives raises AUC from
  0.543 to 0.692 and average precision from 0.107 to 0.204, with no new data.
  The harness now reports `same_type_edge_ratio` and warns when a graph is
  multipartite, so this cannot be misread again.

### Fixed

- **`sample_negatives` crashed on positive endpoints absent from the training
  graph.** networkx returns a `DegreeView` rather than raising for a node it
  does not contain, so degree-matched sampling failed with a confusing
  `TypeError`. Callers that filter cold-start pairs never hit this; callers
  constructing a split directly did.

### Changed

- `docs/Evaluation.md` previously concluded the graph was "too sparse for
  topological link prediction". That was a measurement artefact of using
  length-2 predictors on a multipartite graph, and has been corrected along
  with the guidance that followed from it — a GNN is now worth trying, with
  0.692 as the number it must beat.

### Added

- **Evaluation harness for link prediction** (`litkg.evaluation`, `make
  evaluate`). Splits knowledge graph edges on the publication year of the
  supporting paper, scores structural baselines, and reports AUC, average
  precision, Hits@K and MRR. Documented in `docs/Evaluation.md`.

  The split guards three sources of leakage, each counted in the report: pairs
  re-asserted after the cutoff but first published before it (495 at cutoff
  2016) stay in training; undated gene-variant edges form a backbone that
  disqualifies duplicate test pairs; cold-start pairs with an endpoint missing
  from training are excluded rather than counted as failures.

  Negatives match the endpoint types of the positives they stand against, with
  `--degree-matched` additionally controlling for node popularity.

  **First measured result, and it is negative:** at cutoff 2016 with degree
  matching, Adamic-Adar reaches AUC 0.543 against a random floor of 0.498, and
  Hits@1 is 0.000 for every predictor. Preferential attachment scores 0.725
  without degree matching and 0.512 with it, showing the apparent signal was
  node popularity. The cause is structural: 84.6% of test pairs share no
  neighbour in the training graph, so shared-neighbour predictors are
  undefined for them rather than wrong. The harness reports this as
  `structural_coverage` and warns below 50%.

### Added

- **Clinical entities from CIVIC evidence.** The knowledge graph held only
  genes and variants, so the diseases and chemicals typed NER extracts had
  nothing to link against. CIVIC evidence now contributes 270 diseases (keyed
  by Disease Ontology id where present), 381 therapies and 59 phenotypes, with
  predicates derived from `evidence_type` × `significance` and drawn from the
  same vocabulary the literature extractor emits. Confidence comes from
  CIVIC's evidence level and curator rating rather than a flat 0.8, and the
  498 "Does Not Support" rows are flagged `negated` rather than asserted.
  Cross-modal links rose 92 → 167, now including 51 disease↔disease and 35
  chemical↔drug pairs.

### Fixed

- **4125 of 5825 KG edges dangled.** `_process_civic_evidence` emitted
  relations pointing at `CIVIC:DISEASE:` and `CIVIC:DRUG:` nodes it never
  created, and read three columns the evidence file does not have
  (`variant_id`, `drugs`, `clinical_significance`). Every evidence subject was
  the empty string and no therapy relation was ever built. Evidence subjects
  are now resolved through a molecular-profile index (92.7% direct, the rest
  compound profiles split into components; 1 row of 4254 unresolvable).
- **Overlapping NER spans produced duplicate mentions.** Running two
  specialized models over one text left 109 spans double-tagged with
  disagreeing labels. The first model in `NER_MODELS` now claims a span.
- **Mention keys ignored the entity label,** so a link made against a mention
  read as CHEMICAL landed on the entity resolved as GENE — "BRAF" the gene
  linked to "BRAF Inhibitor" the drug.
- **Entity normalization stripped identity-bearing words.** Removing
  `inhibitor`, `receptor` and `kinase` equated a drug with its target, a
  ligand with its receptor, and turned "anaplastic lymphoma kinase" into
  "anaplastic lymphoma". Only `gene` and `protein` are stripped now.

### Fixed

- **NER entity typing.** Every extracted literature entity was typed `GENE`,
  making all 78 relations GENE→GENE. Two causes compounded: `en_core_sci_md`
  emits a single `ENTITY` label that is not among the kept entity types, so the
  scispacy path returned nothing; the rule-based fallback then accepted any
  all-caps token matching `[A-Z][A-Z0-9]{2,10}` as a gene, so `ALL`, `NSCLC`,
  `PFS`, `DNA` and `ICI` all became genes.

  Extraction now runs `en_ner_bionlp13cg_md` and `en_ner_bc5cdr_md` with their
  labels mapped onto the project's types, and gene acceptance is driven by the
  KG's own gene symbols plus a non-gene acronym stoplist instead of token
  shape. Entities are now 303 DISEASE / 301 GENE / 175 CHEMICAL / 167
  CELL_TYPE / 45 TISSUE / 37 ORGANISM, and GENE→DISEASE is the most common
  relation. Gene recall is unaffected: 66 of the 67 KG gene symbols appearing
  verbatim in the corpus are still extracted.

### Documentation

- Rewrote `README.md` for accuracy. It previously claimed Phases 1–3 complete
  and "90%+ entity extraction accuracy" with no evaluation behind either.
- Added `ARCHITECTURE.md`, `CONTRIBUTING.md`, this changelog,
  `docs/Phase3_README.md` (Phase 3 had no documentation at all),
  `docs/RAG_and_Agents.md`, and `docs/LLM_Setup.md`.
- Replaced fabricated precision figures in `docs/Phase1_README.md` with
  measured counts, and stated why no precision figures are given.
- Removed `docs/LangChain_Integration_Plan.md`, a plan superseded by
  documentation of what actually shipped.

---

## [0.2.0] — 2026-08-17

Two merged pull requests. The suite went from **not collectable** to **241
passing**.

### Fixed — packaging

- **The test suite could not run at all.** `pyproject.toml` set
  `[tool.hatch.build] directory = "src"` — that key sets the build *output*
  directory — alongside `packages = ["litkg"]`, a path that does not exist. The
  editable install produced an empty `.pth`, so `import litkg` failed and all
  189 tests errored during collection. With it fixed, the real baseline was 87
  failed / 101 passed.
- `make run-phase1` was broken twice over: an import of `KGPreprocessor` (the
  class is `KnowledgeGraphPreprocessor`), and a
  `class DisambiguationEngine(...): pass` inserted **into the middle of**
  `EntityLinker`, orphaning three of its methods onto the wrong class.

### Fixed — correctness

- **Four of seven biological plausibility rules were dead.** The lookup sorted
  the type pair while the rule table's keys were unsorted, so every
  gene–disease, drug–disease, protein–disease and pathway–disease pairing fell
  through to the default and was rejected as implausible.
- **`ConfidenceScorer` ignored its own assessors.** It went straight to
  untrained networks, so component assessments had no effect on output.
- **GO IDs were treated as identity evidence.** BRCA1 and BRCA2 both carry
  `GO:0006281` ("DNA repair"), correctly — so they merged into a single entity
  despite distinct CUIs. A GO term annotates function, not identity.
- **BRCA1 and BRCA2 shared a UMLS CUI**, and all three disease UMLS lookups
  were dead (lookup uppercased, keys lowercase).
- `MultiHeadAttention` reused the query's sequence length for key and value, so
  cross-modal attention crashed whenever the two subgraphs differed in node
  count — that is, almost always.
- `ContrastiveLoss` received an alignment matrix sized on node count while
  operating on graph embeddings, and returned `nan` for single-graph batches.
- `MultiTaskLoss` summed into a bare `int`, failing at `.backward()` when no
  task matched.
- `HybridGNNModel` accepted `output_dim`, stored it, and never used it.
- `chunk_overlap` was stored and never used — overlap was genuinely zero, so
  any fact spanning a boundary was unretrievable.
- `OllamaManager.list_models()` parsed a response shape modern `ollama` no
  longer returns, so model discovery silently returned an empty list.
- Failed generations returned `"Error: ..."` as content, making failure look
  like success.
- Three components (`HypothesisGenerator`, `HypothesisValidationAgent`,
  `BiologicalPlausibilityChecker`) only ever tried OpenAI and Anthropic, then
  disabled themselves — permanently LLM-less on a local-only setup.
- `litkg.langchain_integration` was entirely unimportable; LangChain paths had
  moved to `langchain_core`.

### Added — local LLM

- Runs on **Ollama with `qwen3:8b`** by default; no API key required.
- Configurable provider order and model via `config.yaml` / `OLLAMA_HOST` /
  `LITKG_OLLAMA_MODEL`.
- Qwen3 **thinking mode off by default**. With it on, an extraction call spent
  its entire token budget on hidden reasoning and returned **zero characters
  after 128s**; off, the same call answers correctly in ~1s.
- `provider` is now a real parameter on `generate()`; a pinned model or
  provider disables fallback.
- Per-provider option filtering, with Ollama sampling parameters folded into
  its nested `options` dict.

### Added — RAG and agents

- `biomedical_agent` and `rag_system` — both were exported by `__init__.py`,
  headlined in the README, and **never written**.
- `LiteratureRetriever`, `KnowledgeGraphRetriever`, `HybridRetriever` (which
  interleaves rather than concatenates), and `GraphExpansionRetriever`.
- `BiomedicalRAGSystem` generates only from retrieved evidence with `[n]`
  citations, and refuses outright when nothing is retrieved.
- `BiomedicalToolkit`, `BiomedicalQueryAgent`, `HypothesisGenerationAgent`,
  `LiteratureValidationAgent`.

### Added — chunk ↔ graph linkage

- `EntityAliasIndex` and `ChunkGraphIndex` link passages to canonical graph
  nodes, enabling multi-hop retrieval: from one seed passage about BRCA-1,
  expansion walks `BRCA1 → homologous recombination` and surfaces an olaparib
  passage containing no query term at all.

### Added — entity resolution

- Four-rule cascade (identity identifier → normalized name → synonym overlap →
  fuzzy) with union-find for transitivity. `similarity_threshold` now does
  something.
- Surface-form matching runs globally; only the quadratic fuzzy pass uses
  blocking, because synonyms are exactly where surface forms diverge at the
  first character.
- `map_to_umls` now consults loaded ontologies (it never did), any JSON in
  `data/ontologies` auto-loads, and the UMLS REST path works behind
  `UMLS_API_KEY`.
- Seed ontology: 30 terms, 85 surface forms, and deliberately only six CUIs —
  fabricating identifiers would silently merge distinct entities.

### Added — Phase 3

- Uncertainty quantification separating **epistemic** ("unknown") from
  **aleatoric** ("contradictory") via mutual information. This was the
  README's headline claim and had no implementation.
- Platt-scaling confidence calibration.
- Adamic-Adar link prediction requiring no trained embeddings.
- Real PubMed-backed literature validation and temporal trend analysis,
  replacing placeholders.

### Changed

- Chunking: scispacy sentence splitting (the regex broke on "et al.",
  "Fig. 1", "p < 0.05"), token-based sizing capped to the embedding window,
  and section labels carried on chunk metadata.
- `build_networkx_graph` defaults to `MultiDiGraph`; Phase 2 subgraph
  extraction uses `MultiGraph`, since edge features encode `relation_type` and
  collapsing parallel edges hid relations from the model.
- `GNNTrainer` batch handling consolidated from four inconsistent copies into
  one helper.
- `LLMProvider` is a `str` enum and gained `HUGGINGFACE`.

### Removed

- `test_reports/` and the runtime FAISS vector store from version control —
  every test run rewrote tracked files. Only `index.faiss` had ever been
  committed, never the `index.pkl` that `FAISS.load_local` requires, so the
  committed store could not have loaded in a fresh clone anyway.

### Known limits

Carried forward deliberately rather than papered over:

- The ontology rule fires **zero times** on sample data — correct and tested,
  but CIVIC/TCGA records carry no CUIs.
- Entity resolution merges 447 entities, but only ~56 beyond what exact
  matching already caught.
- Literature support classification is a contradiction-cue heuristic over
  title and abstract, not entailment.
- Cross-modal linking yields 100 literature↔KG links on sample data.
- No CI, and no gold-standard evaluation set — which is why this project
  reports counts rather than precision.

---

## [0.1.0]

Initial implementation of Phases 1–3, LangChain integration, and the CLI.
