# LLM Setup

LitKG runs on a local model by default. No API key is required for any feature.

## Quick start

```bash
ollama pull qwen3:8b
```

That is the whole setup. The default configuration already points at `qwen3:8b` on `http://localhost:11434`.

Verify:

```python
from litkg.llm_integration.unified_llm_interface import UnifiedLLMManager

mgr = UnifiedLLMManager()
r = mgr.process_biomedical_task(
    task="entity_extraction",
    input_data="BRCA1 mutations increase breast cancer risk.",
)
print(r.provider.value, r.model, r.content)
```

Expect `ollama qwen3:8b` and an extraction in about a second.

---

## ⚠️ Qwen3 thinking mode

**This is the single most important thing to know about running Qwen3 here.**

Qwen3 is a reasoning model. With thinking enabled it emits hidden reasoning tokens *before* answering, and those tokens count against the generation budget. With a modest `max_tokens`, the entire budget is consumed by reasoning and **the response comes back empty**.

Measured on this project, same prompt and token budget:

| Setting | Time | Output |
|---|---|---|
| `think: true` | 128s | **0 characters** |
| `think: false` | 6.7s | Correct answer |

That is not a slowdown, it is a silent failure that looks like a broken pipeline. Thinking is therefore **off by default**:

```yaml
llm:
  ollama:
    think: false
```

Set `think: true` for genuine multi-step reasoning, and raise `max_tokens` substantially when you do. Extraction and classification tasks do not need it.

---

## Configuration

All settings live in `config/config.yaml` under `llm`:

```yaml
llm:
  # Fallback precedence; first entry is preferred.
  provider_order:
    - "ollama"
    - "anthropic"
    - "openai"

  ollama:
    host: "http://localhost:11434"
    model: "qwen3:8b"
    embedding_model: "nomic-embed-text:latest"
    temperature: 0.2
    max_tokens: 2048
    timeout: 300
    think: false
```

### Environment overrides

| Variable | Overrides |
|---|---|
| `OLLAMA_HOST` | `llm.ollama.host` |
| `LITKG_OLLAMA_MODEL` | `llm.ollama.model` |
| `LITKG_OLLAMA_EMBEDDING_MODEL` | `llm.ollama.embedding_model` |

```bash
LITKG_OLLAMA_MODEL=qwen3-coder:30b make run-langchain
```

## Choosing a model

Any Ollama model works. Registered models carry capability metadata used by `ModelSelector`:

| Model | Size | Notes |
|---|---|---|
| `qwen3:8b` | 5.2 GB | Default. Good reasoning at 8B, long context |
| `qwen3-coder:30b` | 18 GB | Code-tuned; strong structured/JSON output |
| `llama3.1:8b` | 4.9 GB | Solid general baseline |
| `llama3.1:70b` | 42 GB | Highest quality locally, slow without a large GPU |

To use an unregistered model, set `LITKG_OLLAMA_MODEL` — it will be used directly. Registering it in `_load_model_capabilities` additionally lets the selector reason about it.

> `biomedical_score` values in the registry are hand-assigned estimates informed by general capability, **not** benchmark results on biomedical tasks. Treat rankings as a starting point.

## Provider selection

The configured default model wins whenever it satisfies the constraints — an explicit choice is an instruction, not a hint. Without that rule, `llama3.1:70b` would outrank `qwen3:8b` on the capability heuristic and quietly run a 42 GB model.

```python
iface = BiomedicalLLMInterface()

iface.select_best_model_info("literature_analysis")
# -> qwen3:8b (the configured default)

iface.select_best_model_info("literature_analysis", respect_default=False)
# -> llama3.1:70b (pure heuristic ranking)

iface.select_best_model_info("literature_analysis", require_local=True)
iface.select_best_model_info("literature_analysis", available_only=False)  # ignore what is reachable
```

## Pinning and fallback

```python
# Pin a provider; the model is chosen within it
iface.generate(prompt="...", provider=LLMProvider.OLLAMA)

# Pin a model; provider is implied
iface.generate(prompt="...", model="qwen3:8b")

# Mismatch raises rather than silently preferring one
iface.generate(prompt="...", model="gpt-4", provider=LLMProvider.OLLAMA)
# ValueError: Model 'gpt-4' belongs to provider 'openai', not 'ollama'
```

**A pinned model or provider disables fallback.** If you asked for a specific one, silently switching to another defeats the point; you get the real error instead.

Unpinned calls fall back through `provider_order` on failure.

## Cloud providers

Optional. Set the key and the provider becomes available:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export OPENAI_COMPATIBLE_BASE_URL=http://localhost:8000/v1   # vLLM, LocalAI, etc.
```

Reorder `provider_order` to prefer them.

## Generation options

Options are filtered per provider before reaching the SDK. An unrecognized keyword is logged and dropped rather than surfacing as an opaque `TypeError` from inside a client library:

```
Dropping option(s) ['provider_hint'] not supported by ollama; supported: [...]
```

Ollama nests sampling parameters under `options`, but you write them flat and they are folded in:

```python
mgr.process_biomedical_task(task="entity_extraction", input_data="...", top_p=0.9, seed=7)
```

## Troubleshooting

**Empty responses, very slow.** Qwen3 thinking mode. See above.

**"No model satisfies task=..."** — no provider is reachable. Check `ollama list` and that the server is running. `list_models()` returning empty with the server up suggests a client version mismatch; the parser handles both known response shapes.

**Everything falls back to templates.** Components like `HypothesisGenerator` need a working LLM. They try cloud keys, then Ollama, then degrade to templates with a warning in the logs.

**Slow first call.** Ollama loads the model into memory on first use. `keep_alive` controls how long it stays resident.
