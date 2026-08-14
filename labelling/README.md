# Label generation for the KOTE-derived rows

`label_kote_with_gemini.py` is the script that produced the lighting labels attached to the
KOTE-derived text in this project. It is included so the provenance claim in the paper can be
checked rather than taken on trust.

It is reproduced here **verbatim from the upstream working tree** where it was run
(`01_data_prep_labeling_KOTE_v2.py`); only the filename is changed.

## What it does

Reads `KOTE-main/train.tsv`, batches sentences 12 at a time, queries
**`gemini-2.5-flash-lite`** with a structured prompt, and appends the parsed
`sentence,label,h,s,b,dimmer,ct` rows to `labeled_kote_results.csv`. It is resumable: it
counts the rows already written and continues from there, with retries and batch splitting on
failure.

The API key is read from the `GEMINI_API_KEY` environment variable and is never stored in the
file.

```bash
export GEMINI_API_KEY=...
python label_kote_with_gemini.py      # expects KOTE-main/train.tsv alongside it
```

## Provenance chain

The labels used throughout this project trace as follows. The evidence is stated so a reader
can judge it, not just accept it:

| Step | Artifact | Evidence |
|---|---|---|
| Source text | KOTE `train.tsv` (MIT, see `../THIRD_PARTY_LICENSES.md`) | — |
| Labelling | this script, `gemini-2.5-flash-lite` | `MODEL_NAME` at line 20 |
| Raw output | `labeled_kote_results.csv`, 40,000 rows | header `sentence,label,h,s,b,dimmer,ct` matches this script's writer exactly; an earlier superseded script wrote a different schema |
| Splits | `train_kote.jsonl` (36,000), `val_kote.jsonl` (2,000), `test_kote.jsonl` (2,000) | 36,000 + 2,000 + 2,000 = 40,000, matching the CSV row count and the recorded split sizes |

Those three splits enter this repository as the rows tagged `base_train`, `base_val` and
`base_test`. The `base_test` rows are re-tagged as `natural_language_baseline` when the
benchmark set is assembled, so **the natural-language evaluation segment carries these same
Gemini-generated labels** — it is not separately labelled.

## Limitations

1. **No per-row generation log was kept.** The chain above is established from the script, the
   output schema and the row counts. Which specific API call produced a given row cannot be
   recovered from anything released here.
2. **An earlier labelling script existed** upstream (`gemini-1.5-flash`, written one day
   before this one) and wrote to the same output filename in append mode. The CSV's schema
   matches this script and not that one, but a mixed-provenance file cannot be excluded from
   timestamps alone.
3. **`KOTE-main/train.tsv` is not redistributed here.** Obtain it from the upstream KOTE
   release; its license notice is in `../THIRD_PARTY_LICENSES.md`.
4. The script's comments and its error messages are in Korean, as written. They are left
   unchanged so the file matches what was actually executed.

## Why this matters for reading the results

Because these labels are model-generated, any comparison between this project's models and
the labelling model is a **teacher–student** comparison and must be described as such. The
separate, deterministic policy labels used for the rule-based evaluation segment are produced
by `backend/models/compact_llm/training_data.py` and have nothing to do with this script.
