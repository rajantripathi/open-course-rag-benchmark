# Open Course RAG Benchmark

Fully open benchmark and pipeline for multilingual retrieval-grounded educational question answering in higher education.

## Scope

This repository targets a paper and benchmark built from:

- `OpenStax Principles of Data Science`
- `MIT OpenCourseWare` ethics course materials

The benchmark is designed for:

- English and Uzbek questions
- BM25, dense, and hybrid retrieval
- evidence-grounded answer generation
- fully open data and model dependencies

## Repository Layout

```text
configs/                 Retrieval and generation configuration
data/
  raw/                   Download scripts and provenance notes
  processed/             Documents and chunk outputs
  benchmark/             Questions, labels, and annotation templates
docs/                    Benchmark and annotation documentation
paper/                   Manuscript support files
results/
  figures/               Paper-ready plots
  tables/                Paper-ready tables
scripts/isambard/        Deploy and remote execution helpers
slurm/                   Isambard batch scripts
src/open_course_rag_benchmark/
tests/
```

## Data Contracts

### `documents.jsonl`

```json
{
  "doc_id": "ds_ch03",
  "course_id": "openstax_data_science",
  "title": "Introduction to Statistical Analysis",
  "source_type": "textbook",
  "license": "CC BY 4.0",
  "source_url": "https://openstax.org/...",
  "download_date": "2026-04-13",
  "text": "..."
}
```

### `chunks.jsonl`

```json
{
  "chunk_id": "ds_ch03_007",
  "doc_id": "ds_ch03",
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
  "course_id": "mit_ethics",
  "language": "uz",
  "question_type": "conceptual",
  "question_text": "Algoritmik tarafkashlik nima?",
  "reference_answer": "..."
}
```

### `gold_labels.jsonl`

```json
{
  "qid": "Q042",
  "gold_doc_ids": ["ethics_ai_bias_notes"],
  "gold_chunk_ids": ["ethics_ai_bias_004"]
}
```

### `answers.jsonl`

```json
{
  "qid": "Q042",
  "system": "hybrid",
  "retrieved_chunk_ids": ["ethics_ai_bias_004", "ethics_ai_bias_006"],
  "answer": "...",
  "abstained": false
}
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
pytest -q
```

Example local pipeline on the included fixtures:

```bash
ocrb ingest --input-dir tests/fixtures/raw_docs --output data/processed/documents.jsonl
ocrb chunk --documents data/processed/documents.jsonl --output data/processed/chunks.jsonl
ocrb bm25 --chunks data/processed/chunks.jsonl --questions tests/fixtures/questions.jsonl --output results/tables/bm25.jsonl
ocrb eval-retrieval --retrieval results/tables/bm25.jsonl --gold tests/fixtures/gold_labels.jsonl --questions tests/fixtures/questions.jsonl --output results/tables/bm25_metrics.json
```

## Isambard Workflow

Raw course materials and heavy outputs stay on scratch:

```text
$SCRATCH/open-course-rag-benchmark/
```

Typical flow:

```bash
bash scripts/isambard/deploy.sh
sbatch slurm/00_setup.sh
sbatch slurm/01_ingest.sh
sbatch slurm/02_chunk.sh
```

## Notes

- Commit scripts and benchmark artifacts, not large raw course dumps.
- Record provenance for every source document.
- Gold labels must be manually verified.

