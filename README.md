# The Attachment Index

---

## Overview

This repository contains the implementation used for the experiments in the paper. The framework probes attachment-related language in LLM dialogue using two evaluation protocols:

- **IAB (Inferred Attachment Behaviour)** — administer an Adult Attachment Interview (AAI) directly to the target LLM.
- **IDB (Induced Dialogue Behaviour)** — first run a persona-driven conversation between a UserLLM and the target LLM, then administer the AAI to the target LLM with that dialogue as context.

For both protocols, a Judge LLM scores the resulting AAI transcript along narrative and linguistic dimensions and assigns one of: `secure`, `dismissive`, `fearful`, `anxious`, or `undefined`.

Supported providers (via `src/llm_attachment_index/llm_calls.py`):

- **Closed/frontier (OpenAI-compatible APIs):** OpenAI, Anthropic, Google (Vertex AI OpenAPI endpoint), DeepSeek, OpenRouter
- **Open weights:** any HuggingFace `text-generation` model (loaded locally via `transformers`)
- **Mock:** a `MockLLM` for offline dry runs

---

## Method Overview

### 1. Direct AAI-style questioning (`iab`)

The target (primary) LLM is asked the 14 AAI questions in `AAIQuestions.QUESTIONS` (see `src/llm_attachment_index/llm_agents.py`). The transcript is then evaluated by the Judge LLM against two rubrics:

- `AAIEvaluationSchema` — narrative dimensions (coherence, emotional expression, attitude toward caregivers, reflective function, response length).
- `AAILinguisticSchema` — Grice's maxims (quality, quantity, relevance, manner).

### 2. Persona-based interaction (`idb1`, `idb2`, `idb3`)

A `HumanLLMAgent` is instantiated from one of the auto-generated personas (a cross-product of `Demographics` traits in `constants.py`, paired with a summarised real-world issue sampled from the bundled CAMS / ESConv datasets). It opens a multi-turn conversation (10–20 turns, randomised; see `conversation.py`) with the primary LLM, steered by one of four attachment styles: `secure`, `dismissive`, `fearful`, `anxious`. The three IDB modes vary how strongly the persona reveals its style:

| Run    | Scenario                                                        |
| ------ | --------------------------------------------------------------- |
| `idb1` | Neutral — attachment style is **not** revealed                  |
| `idb2` | Implicit — style is hinted via actions/cues                     |
| `idb3` | Explicit — style is openly discussed and tied to the user issue |

After the conversation, the AAI is administered to the primary LLM with the dialogue as prior context.

---

## Installation

This project uses Poetry for dependency management and requires Python `^3.11`.

### 1. Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Clone the repository

```bash
git clone https://github.com/animeshprasad/llm_attachment_index.git
cd llm_attachment_index
```

### 3. Install dependencies

```bash
poetry install
```

The lockfile pins `openai`, `transformers`, `torch`, `pandas`, `scikit-learn`, and `scipy`. HuggingFace models also use `huggingface_hub` for login (installed transitively with `transformers`).

### 4. Activate the virtual environment

```bash
poetry shell
```

---

## Configuration

All models, providers, and credentials are declared in `src/llm_attachment_index/config.json`. There is no `.env` loading; API keys live in this file (so do **not** commit real keys — keep a local copy out of version control).

The config schema is `{"models": {"<alias>": {"provider": ..., "model": ..., "api_key": ..., ...}}}`. Aliases are what you pass to `--primary`, `--human`, and `--judge`.

> **The shipped `config.json` is a dummy template.** It only documents the schema and the providers the code supports — the entries inside it use placeholder API keys (e.g. `"sk-your-hf-key"`) and are **not runnable as-is**. Before you run anything other than `--dev`, edit the file to:
>
> 1. Define a `gpt-cheap` alias (see [Required alias: `gpt-cheap`](#required-alias-gpt-cheap) below) — without it, every `idb*` run will `KeyError`.
> 2. Fill in real `api_key`s (and `project_id`/`location` for Google) on every alias you actually intend to use.
> 3. Add aliases for whatever model you pass to `--primary`, plus the aliases referenced by `run_experiments.sh` if you want to use the batch script.
>
> If you just want to smoke-test the pipeline with zero API calls, point every alias you need at the mock provider, e.g.:
>
> ```jsonc
> "gpt-cheap": { "provider": "mock", "model": "mock-model", "api_key": "mock-key" }
> ```

Supported `provider` values and the extra fields they expect:

| Provider      | Required fields                              | Notes                                                                                 |
| ------------- | -------------------------------------------- | ------------------------------------------------------------------------------------- |
| `openai`      | `api_key`, `model`                           | Standard OpenAI Chat Completions endpoint                                             |
| `anthropic`   | `api_key`, `model`                           | Called via the OpenAI-compatible base URL                                             |
| `deepseek`    | `api_key`, `model`                           | OpenAI-compatible                                                                     |
| `google`      | `api_key`, `model`, `project_id`, `location` | Vertex AI OpenAPI endpoint (`location` defaults to `us-central1`)                     |
| `openrouter`  | `api_key`, `model`                           | OpenAI-compatible                                                                     |
| `huggingface` | `api_key`, `model`                           | Loaded locally with `transformers.pipeline("text-generation", ...)` — needs a HF token |
| `mock`        | `api_key`, `model`                           | Returns lorem-ipsum; used by `--dev` and for smoke-testing                            |

### Required alias: `gpt-cheap`

`gpt-cheap` is the canonical "cheap utility call" alias. It is the default for `--judge` and `--human`, **and** is hard-coded inside `experiment.py` for the bare LLM that summarises CAMS / ESConv samples in `PersonaMetadata.generate_all_personas`. **Every `idb*` run requires it**, even when `--judge`/`--human` are overridden on the CLI.

For real runs, point it at a cheap, fast, instruction-following chat model (e.g. `gpt-4o-mini`, Anthropic Haiku, an OpenRouter route, or a local HuggingFace model):

```jsonc
"gpt-cheap": {
  "provider": "openai",
  "model": "gpt-4o-mini",
  "api_key": "sk-..."
}
```

For smoke tests (no API calls):

```jsonc
"gpt-cheap": {
  "provider": "mock",
  "model": "mock-model",
  "api_key": "mock-key"
}
```

### Adding more models

Add further aliases for the models you want to audit (`--primary`) or use as the human/judge. A minimal runnable config looks like:

```jsonc
{
  "models": {
    "gpt-cheap":    { "provider": "openai",    "model": "gpt-4o-mini",                "api_key": "sk-..." },
    "my-primary":   { "provider": "anthropic", "model": "claude-3-5-sonnet-latest",   "api_key": "sk-ant-..." }
  }
}
```

You can point the runner at a different config file with `--config /path/to/config.json`.

---

## Running Experiments

The entry point is `src/llm_attachment_index/experiment.py`. All CLI flags are defined in `parse_args()` (see `src/llm_attachment_index/utils.py`):

| Flag                 | Default                                | Description                                                                          |
| -------------------- | -------------------------------------- | ------------------------------------------------------------------------------------ |
| `--run`              | `iab1`                                 | One of `iab`, `idb1`, `idb2`, `idb3`                                                 |
| `--primary`          | `gemini`                               | Alias of the target LLM (must exist in `config.json`)                                |
| `--human`            | `gpt-cheap`                            | Alias of the UserLLM that drives the persona (IDB only)                              |
| `--judge`            | `gpt-cheap`                            | Alias of the Judge LLM                                                               |
| `--strong_priming`   | off                                    | Add a system-prompt instruction telling the primary LLM to act as a human companion |
| `--tapered_response` | on                                     | Pre-fill the AAI answer with a tapering string (`"I feel  "` by default)             |
| `--tapering_string`  | `"I feel  "`                           | The string used for tapering                                                         |
| `--dev`              | off                                    | Force `primary`/`judge`/`human` to the `mock` provider (no API calls)                |
| `--config`           | `src/llm_attachment_index/config.json` | Path to the model config file                                                        |

> **`gpt-cheap` is required for IDB runs.** It is used as the default for `--judge` and `--human`, *and* is hard-coded inside `experiment.py` as the "bare" model that summarises CAMS / ESConv samples into each persona's `My issue` text (`config["models"]["gpt-cheap"]` in `PersonaMetadata.generate_all_personas`). Even if you override `--judge` and `--human` on the CLI, any `idb*` run will `KeyError` if `gpt-cheap` is missing from `config.json`. The shipped config defines it as a cheap OpenAI chat model — point it at whatever cheap, fast, instruction-following model you want for these utility calls (e.g. `gpt-4o-mini`).
>
> The `--primary` default (`gemini`) and the aliases used by `run_experiments.sh` (`mistral`, `gemini`, `llama`, `deepseek`) are **not** shipped — either pass `--primary <alias>` explicitly or add matching entries pointing at the models you want to audit.

### Direct AAI (`iab`)

```bash
poetry run python src/llm_attachment_index/experiment.py \
  --run iab \
  --primary openai-o3-mini-2025-01-31 \
  --judge gpt-cheap \
  --strong_priming
```

### Persona-based interaction (`idb1` / `idb2` / `idb3`)

For IDB runs, personas are generated from `PersonaMetadata.generate_all_personas` (cross-product of `Demographics` core traits) and each persona is paired with **one** issue sampled from the bundled CAMS / ESConv datasets and summarised by the model aliased as `gpt-cheap`. There is no `--persona` flag.

> **Cost warning — IDB invocations fan out aggressively.** A single `idb*` call iterates `len(GENDER) × len(AGE_GROUP)` personas (12 with the shipped `Demographics`) × `len(InteractionScenarios.attachment_style)` attachment styles (4) = **48 conversations + AAI interviews + judge calls per invocation**, each with its own sampled CAMS/ESConv issue. Costs and rate limits will be the dominant constraint. If you only want a smoke test or a cheap pilot, add a temporary `break` after the inner `for attachment_index in range(...)` loop in the `elif args.run.startswith('idb'):` block of `src/llm_attachment_index/experiment.py` to cap at one persona × 4 attachment styles (= 4 runs per invocation), and remove it again when you want the full sweep.

```bash
poetry run python src/llm_attachment_index/experiment.py \
  --run idb2 \
  --primary anthropic-claude-3-7-sonnet-thinking \
  --human gpt-cheap \
  --judge gpt-cheap \
  --strong_priming
```

### Dry run (no API calls)

```bash
poetry run python src/llm_attachment_index/experiment.py --run iab --dev
```

### Batch script

`run_experiments.sh` sweeps a set of models × experiment types. It currently references the aliases `mistral`, `gpt-cheap`, `gemini`, `llama`, `deepseek`, so update either the script or your `config.json` so the aliases match before running it.

```bash
./run_experiments.sh
```

### Re-running just the judge on existing results

`run_judge_only.py` re-evaluates every JSON file in `results/` with a fresh Judge LLM. Edit the `judge_config` block at the top of the file to point at your model and key, then:

```bash
poetry run python src/llm_attachment_index/run_judge_only.py
```

### Aggregating results

`evaluate_results.py` aggregates all `results/*.json` into per-model / per-experiment / per-attachment-style pivot tables (split by primed vs unprimed):

```bash
poetry run python src/llm_attachment_index/evaluate_results.py
```

---

## Repository Structure

```text
llm_attachment_index/
├── pyproject.toml                # Poetry config (Python ^3.11)
├── poetry.lock
├── run_experiments.sh            # Bash sweep over models × experiment types
├── LICENSE                       # MIT
├── src/
│   └── llm_attachment_index/
│       ├── __init__.py
│       ├── config.json           # Model + provider + API key definitions
│       ├── constants.py          # Demographics, persona generation, CAMS/ESConv loader
│       ├── conversation.py       # IDB scenario prompts + multi-turn dialogue loop
│       ├── experiment.py         # CLI entry point: runs IAB and IDB evaluations
│       ├── llm_agents.py         # LLMAgent / HumanLLMAgent / JudgeLLMAgent + AAI rubrics
│       ├── llm_calls.py          # Provider wrappers (OpenAI, HF, mock, etc.)
│       ├── utils.py              # Argparse, validation, model cache, aggregation helpers
│       ├── evaluate_results.py   # Aggregate JSON results into pivot tables
│       ├── run_judge_only.py     # Re-score existing results with a different judge
│       ├── data/
│       │   ├── added_CAMS_data.csv   # CAMS subset used for persona issue grounding
│       │   └── ESConv.json           # ESConv dialogues used for persona issue grounding
│       └── annotations/
│           ├── analyse_annotations.py
│           ├── generate_annotation_html.py
│           └── assets/
└── tests/
    ├── test_interaction_scenarios.py
    └── test_persona_generation.py
```

---

## Outputs

Each experiment writes a single JSON file to `results/` named `<exp_type>_<hash>.json`, where `<hash>` is the first 8 hex chars of an MD5 over the parameter dict (so re-running the same configuration is cached and skipped). A sibling `results/experiment_mapping.json` maps hashes back to the full parameter set.

```text
results/
├── experiment_mapping.json       # {hash: params}
├── iab_<hash>.json               # one per (primary, judge, priming, tapering)
└── idb1_<hash>.json | idb2_<hash>.json | idb3_<hash>.json
                                  # one per (primary, human, judge, persona, attachment_style, ...)
```

Each result file contains, at minimum:

```jsonc
{
  "evaluation_type": "idb2",
  "primary_model": "...",
  "human_model": "...",          // IDB only
  "judge_model": "...",
  "persona": "...",              // IDB only
  "attachment_type": "anxious",  // IDB only; the induced style
  "conversation_history": [ ... ], // pre-AAI dialogue (IDB) or [] (IAB)
  "scoring_pairs": [ ["question", "response"], ... ],
  "narrative_judgment": "Narrative_Coherence: ...\nOverall: ...",
  "narrative_attachment_type": "secure | dismissive | fearful | anxious | undefined",
  "linguistic_judgment": "Quality: ...\nOverall: ...",
  "linguistic_attachment_type": "secure | dismissive | fearful | anxious | undefined",
  "strong_priming": true
}
```

---

## Development

Run the test suite:

```bash
poetry run pytest
```

Format and lint:

```bash
poetry run black .
poetry run flake8
```

---

## Citation

```bibtex
@inproceedings{demeocq2026attachment,
  title={The Attachment Index: Auditing Attachment Language Cues and Relational Safety Risks in Human-LLM Dialogue},
  author={Demeocq, Cyndie and Prasad, Animesh and Saeidi, Marzieh and Goodall, Karen and Ross, Bj\"orn},
  booktitle={Proceedings of the CLPsych Workshop at ACL},
  year={2026}
}
```

---

## License

MIT License — see [`LICENSE`](LICENSE).
