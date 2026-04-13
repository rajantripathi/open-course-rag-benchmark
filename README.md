# Open Course RAG Benchmark

Open multilingual benchmark and pipeline for retrieval-grounded educational question answering in higher education.

## Scope

This repository targets a paper and benchmark built from:

- `OpenStax Principles of Data Science` (`CC BY-NC-SA 4.0`)
- `OpenStax Introduction to Philosophy` (`CC BY 4.0`)

The benchmark is designed for:

- English and Uzbek questions
- BM25, dense, and hybrid retrieval
- evidence-grounded answer generation
- fully open source data, code, and models

The benchmark release uses `CC BY-NC-SA 4.0`, the more restrictive of the two
source licenses.

## Data Statement

Both OpenStax books contain the clause: "This book may not be used in the
training of large language models." This project uses the books as a static
retrieval corpus for evaluation only. The texts are not used to train or
fine-tune any model weights.

## Repository Layout

```text
configs/                 Retrieval, generation, and chunking configuration
data/
  raw/                   Provenance notes and download manifests
  processed/             Lightweight processed outputs
  benchmark/             Questions, labels, and manual review artifacts
docs/                    Benchmark and annotation documentation
paper/                   Manuscript support files
results/
  figures/               Paper-ready figures
  tables/                Paper-ready tables
scripts/                 Local review/export helpers
scripts/isambard/        Deploy and remote execution helpers
slurm/                   Isambard batch scripts
src/open_course_rag_benchmark/
tests/
```

## Data Contracts

### `documents.jsonl`

```json
{
  "doc_id": "ds_ch03_s02",
  "course_id": "openstax_data_science",
  "title": "3.2 Statistical Measures",
  "source_type": "textbook_section",
  "license": "CC BY-NC-SA 4.0",
  "source_url": "https://openstax.org/books/principles-data-science/pages/3-2-statistical-measures",
  "download_date": "2026-04-14",
  "text": "..."
}
```

### `chunks.jsonl`

```json
{
  "chunk_id": "ds_ch03_s02_007",
  "doc_id": "ds_ch03_s02",
  "course_id": "openstax_data_science",
  "section": "3.2 Statistical Measures",
  "chunk_index": 7,
  "text": "..."
}
```

### `questions.jsonl`

```json
{
  "qid": "Q042",
  "course_id": "openstax_philosophy",
  "language": "uz",
  "question_type": "conceptual",
  "question_text": "Epistemologiya nimani o'rganadi?",
  "reference_answer": "..."
}
```

### `gold_labels.jsonl`

```json
{
  "qid": "Q042",
  "gold_doc_ids": ["phil_ch02_s04"],
  "gold_chunk_ids": ["phil_ch02_s04_004"]
}
```

### `answers.jsonl`

```json
{
  "qid": "Q042",
  "system": "hybrid",
  "language": "uz",
  "retrieved_chunk_ids": ["phil_ch02_s04_004", "phil_ch02_s04_006"],
  "answer": "...",
  "abstained": false
}
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
```

## Isambard Workflow

Scratch root:

```text
/scratch/u6ef/rajantripathi.u6ef/open-course-rag-benchmark
```

Typical flow:

```bash
bash scripts/isambard/deploy.sh
sbatch slurm/00_setup.sh
sbatch slurm/00_scrape.sh
sbatch slurm/01_ingest.sh
sbatch slurm/02_chunk.sh
```

## Notes

- Commit scripts, metadata, benchmark artifacts, and final tables/figures.
- Do not commit large raw textbook dumps or vector indexes.
- Gold labels and grounding labels must be manually verified.
- Primary journal target: `Education and Information Technologies`.

