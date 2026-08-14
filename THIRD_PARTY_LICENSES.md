# Third-party licenses

This release includes material derived from third-party work. Their license terms are
reproduced below and continue to apply to the derived material.

---

## KOTE (Korean Online That-gul Emotions)

Text from the KOTE dataset is present in this repository in derived form. It appears in:

- `backend/models/compact_llm/datasets/pretrain_runs/20260805_clean_v1/pretrain_corpus.txt`
  and the accompanying token report
- `backend/models/compact_llm/datasets/scale_grid_subsets/*.jsonl` and
  `datasets/policy_only/*.jsonl` — rows whose `source` field is `base_train`, `base_val`
  or `base_test`
- `backend/qwen_service/training/data/*.jsonl`
- `backend/reports/benchmark_datasets/v3_test_mixed_2000.jsonl` — the `natural_language_baseline`
  segment
- the per-row prediction files, which quote the corresponding input text

The lighting values attached to those rows are this project's own labels, produced as
described in the README; the input text is KOTE's.

**License: MIT**

```
MIT License

Copyright (c) 2022 Jeon Duyoung

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

MIT permits redistribution and modification provided this notice travels with the material,
which is why it is reproduced here. It is compatible with this repository's Apache-2.0
license and with the CC-BY-4.0 license of the companion checkpoint record.

### A limitation worth stating

The labels carried on the original KOTE-derived rows (`source: base_*`) came with the source
data. This repository does not record whether those particular labels were produced by human
annotators, by a rule, or by a model, and that cannot be established from the code here. The
project's own labels — the deterministic policy outputs and the teacher-model labels for the
natural-language segment — are documented in the README and are separate from these.

---

## Qwen3-0.6B

The comparator is a LoRA fine-tune of Qwen3-0.6B-Base. Base model weights are not
redistributed here; only the LoRA adapter and the training code are included. The base model
is subject to its own license from Alibaba Cloud, which applies to any use of the merged or
quantized comparator.
