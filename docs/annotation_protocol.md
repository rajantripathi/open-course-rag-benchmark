# Annotation Protocol

## Question Selection Protocol

1. Load all candidates from `candidates.jsonl`.
2. For each course, select exactly 60 questions:
   - 20 factual
   - 20 conceptual
   - 10 procedural
   - 10 comparative
3. Selection criteria:
   - Question is unambiguous and answerable from the source corpus
   - Question does not require external knowledge beyond the textbook
   - Question is not trivially simple
   - Question covers diverse chapters
4. For each selected question, verify and correct the reference answer.

## Gold Evidence Assignment Protocol

1. For each selected question, identify the chunk(s) that contain sufficient evidence to answer it.
2. Assign `gold_doc_ids` and `gold_chunk_ids`.
3. A question may have 1-3 gold chunks.
4. If no single chunk set fully answers the question, revise or drop the question.

## Uzbek Translation Review Protocol

1. For each Uzbek translation, verify:
   - semantic equivalence with English original
   - natural Uzbek phrasing
   - technical terms handled appropriately
2. Correct any mistranslations.

## Groundedness Labels

Allowed labels:

- `correct_grounded`
- `partially_correct`
- `unsupported`
- `hallucinated`

## Intra-Annotator Agreement

- Re-annotate at least 20 answers after a delay.
- Compute Cohen's kappa on duplicate pairs.

