# Annotation Protocol

## Question Curation

- Generate candidate questions from source sections with an LLM if needed.
- Keep only questions answerable from the course materials themselves.
- Maintain balance across:
  - course
  - language
  - question type

## Gold Evidence Assignment

- Assign `gold_chunk_ids` manually.
- Prefer the smallest chunk set that fully supports the answer.
- If multiple chunks are essential, include all required supporting chunks.
- Do not use model-generated evidence labels without human review.

## Groundedness Labels

Use one label per answer:

- `correct_grounded`
- `partially_correct`
- `unsupported`
- `hallucinated`

## Intra-Annotator Agreement

- Re-annotate at least 20 answers after a delay.
- Compute Cohen's kappa between the first and second passes.

