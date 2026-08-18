# Contributing

## Setup

```bash
git clone https://github.com/poglesbyg/LitKG.git
cd LitKG
make install-dev          # dependencies + dev tooling
uv run python scripts/setup_models.py   # scispacy biomedical models
ollama pull qwen3:8b      # local LLM, no API key needed
```

> `uv sync` prunes the scispacy models, since `setup_models.py` installs them
> outside the lockfile. If sentence splitting quietly degrades to regex, re-run
> that script.

## Tests

```bash
make test                 # full suite, ~45s
uv run pytest tests/test_phase1.py -v
uv run pytest -k "entity_resolution" -v
uv run pytest --no-cov    # skip coverage for a faster loop
```

**276 tests currently pass**, and CI runs them on every push and pull request.
Run them locally before pushing anyway — the suite takes about 45 seconds, and
finding out from CI is slower than finding out from your terminal.

CI gates on three things: every module imports, `flake8` finds no syntax errors
or undefined names, and the suite passes. Style and type findings are reported
but do not block, because the codebase carries substantial pre-existing debt
(~2.7k style, ~487 type). Paying that down is welcome; the gate can tighten as
it shrinks.

CI runs Python 3.11 only. scispacy pins `thinc <8.2.0`, and thinc 8.1.12 has no
3.12 wheel — it tries to compile from source and fails in Cython.

Tests must not require network access or an LLM. Inject stubs instead:

```python
def test_answer_is_grounded(stub_llm_manager):
    rag = BiomedicalRAGSystem(knowledge_graph=graph, llm_manager=stub_llm_manager)
```

Live-service checks belong in a script under `scripts/`, not the suite.

## Code style

```bash
make format               # black + isort
make lint                 # flake8
make typecheck            # mypy
```

Line length 100. Match the surrounding file's conventions over any global rule.

## Conventions that matter here

These are the ones the codebase has been burned by. They are worth following.

### Fail loudly at boundaries

A failed operation must not return a value that looks like success:

```python
# Wrong: a failure becomes an LLMResponse whose content reads "Error: ..."
except Exception as e:
    return f"Error: {e}"

# Right
except Exception as e:
    self.logger.error(f"Generation failed for {model}: {e}")
    raise
```

**One exception:** LangChain tool functions. There, returning error text lets
the agent read it and recover, while raising aborts the whole run. Those sites
are commented as such — do not "fix" them.

### Normalize both sides of a lookup

Two bugs in this codebase came from the same mistake — a lookup normalized one
way and keys stored another:

```python
# The rule table's keys were unsorted; the lookup sorted the pair.
# Four of seven rules could never match.
self.plausibility_rules = {tuple(sorted(pair)): score for pair, score in {...}.items()}
```

If you normalize for lookup, normalize at construction too.

### Do not fabricate identifiers

A wrong UMLS CUI is worse than a missing one. Entity resolution treats a shared
CUI as **decisive**, so an invented collision silently merges two distinct
entities while looking authoritative. Leave the field empty and let coverage
come from a licensed source.

The seed ontology generator asserts CUI uniqueness for this reason.

### Know what identifies versus what describes

A GO term annotates what an entity *does*, not which entity it *is*. BRCA1 and
BRCA2 both carry `GO:0006281` ("DNA repair"), correctly. Only identifiers in
`IDENTITY_IDENTIFIERS` may drive merging.

### Do not silently discard information

Where information must be dropped — flattening a `MultiDiGraph` for an
algorithm that needs a simple graph, dropping an unsupported provider option —
do it explicitly and say why:

```python
if dropped:
    self.logger.warning(f"Dropping option(s) {dropped} not supported by {provider.value}")
```

### Parameters must do something

Several parameters in this codebase were accepted, stored, and never read:
`chunk_overlap` (overlap was actually zero), `output_dim` (no layer produced
it), `similarity_threshold` (matching was exact-only). If you add a parameter,
use it or do not accept it.

### Aliases, not duplicated state

Renamed fields are read-only properties or `__post_init__`-synced pairs, so
there is one source of truth:

```python
@property
def head_entity(self) -> str:
    """Alias for entity1, the subject of the relation."""
    return self.entity1
```

## Docs

Update docs in the same change as the code. Specifically:

- New public API → the relevant `docs/*.md`
- Behaviour change → check `README.md` and `ARCHITECTURE.md`
- Anything user-visible → `CHANGELOG.md`

**State limits plainly.** Several docs previously quoted precision figures with
no evaluation behind them. If a number is not measured, do not print it; if a
capability is a heuristic, say so.

## Pull requests

Branch from `main` — do not commit to it directly.

```bash
git checkout -b fix/short-description
make test
git push -u origin fix/short-description
gh pr create
```

A useful PR description says what was broken, why it mattered, and what a
reviewer should be skeptical of. If you edited a test, explain why the test was
wrong rather than the code.

## Open needs

Genuinely useful contributions, roughly in order:

1. **Ontology coverage.** Entity resolution's strongest rule is inert without
   CUIs. A UMLS loader, or a larger curated vocabulary, unblocks it.
3. **Cross-modal linking.** 100 literature↔KG links on sample data, up from 12
   once gene-level nodes existed. `EntityLinker` still has not had the attention
   KG-internal resolution has.
4. **Support classification.** Literature validation uses contradiction cues on
   title and abstract. Entailment over full text would be a real improvement.
5. **Evaluation set.** There is no gold standard, which is why this project
   reports counts rather than precision.
