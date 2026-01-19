# Self-Instruct and Auto Evol-Instruct

Generate analytical questions about customer-support conversations without using transcripts as input. The pipeline first expands seed questions, then rewrites them with additional constraints or reasoning requirements.

## Setup

Run from this directory so `.env`, seed files, and output paths resolve correctly:

```bash
cd queries-generation/method-2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `GROQ_API_KEY` in your environment or in a local `.env` file:

```dotenv
GROQ_API_KEY=your_key_here
```

Configuration validates the key during import, before command-line arguments are parsed. Set the environment variable or `.env` value before running any script; `--api-key` alone cannot bypass that check.

The pipeline also loads a sentence-transformer model and NLTK stopwords. The first run may download these resources.

## Run

Generate seeds before starting the pipeline:

```bash
python seed_generator.py
python main.py --stage all
```

To run stages separately:

```bash
python main.py --stage 1
python main.py --stage 2
```

Stage 2 reads the saved stage 1 output. To skip its prompt-optimization cycle:

```bash
python main.py --stage 2 --no-optimization
```

These commands make Groq API requests. Output counts depend on generation and filtering results.

## Outputs

| File | Contents |
| --- | --- |
| `output/generated_seeds.json` | Seed questions grouped by domain |
| `output/self_instruct_queries.json` | Stage 1 questions, metadata, and diversity metrics |
| `output/evolved_queries.json` | Stage 2 rewrites and evolution results |

Rerunning a stage writes to the same configured output path. Copy any results you want to retain before starting another run.

## Configuration and source files

| File | Purpose |
| --- | --- |
| [`config.py`](config.py) | Models, domains, generation limits, difficulty distribution, and output paths |
| [`seed_generator.py`](seed_generator.py) | Generate and save initial questions |
| [`self_instruct.py`](self_instruct.py) | Expand seeds, filter duplicates, and calculate diversity metrics |
| [`auto_evol.py`](auto_evol.py) | Rewrite questions and optimize the evolution method |
| [`utils/failure_detector.py`](utils/failure_detector.py) | Heuristics for rejecting unsuccessful rewrites |
| [`utils/prompts.py`](utils/prompts.py) | Prompt templates |
| [`utils/llm_client.py`](utils/llm_client.py) | Groq requests and retry handling |

The configured domains are Hotel, Flight, Retail, Banking, Telecom, and Insurance. Change `DOMAINS` in `config.py`, then regenerate seeds to use a different set. Model names and sampling settings are defined in `MODEL_CONFIG`.

## Limitations

Generated questions need review for answerability against the intended transcripts. Diversity metrics and rewrite filters do not establish factual accuracy or retrieval quality.

This directory does not contain the standalone DeepEval evaluation scripts described in earlier documentation. No runtime, cost, or quality benchmarks are asserted here. For the separate method 1 evaluation notebooks, see [query evaluation](../method-1/query-evaluation/README.md).
