# 05 · Contamination screen and calibration sensitivity analyses

Supports Sections 4.3 and 4.6 of the manuscript (response to Reviewer 1, Comment 15).

## Files
| File | Purpose |
|---|---|
| `contamination_check.py` | Word 13-gram screen of all 7,045 evaluation questions against the fine-tuning corpus (as in the earlier MediLITE-QA study). Writes the exclusion masks. |
| `overlap_sensitivity.py` | Recomputes accuracy, ECE (15 bins), AUROC and AURC with 95% item-bootstrap intervals (2,000 resamples, seed 1234) on: all questions; all minus the flagged questions; MedMCQA/MedQA/PubMedQA minus flagged; the six MMLU subsets only. Reuses the functions of `../01_calibration_cascade/calibration_cascade_pilot.py`. |

## Exclusion masks (`results/05_contamination_sensitivity/contamination_out/`)
- `contaminated_items.json`: 40 dirty questions (coverage >= 80% or duplicated question stem), `{task: [doc_id, ...]}`
- `not_clean_items.json`: 344 not-clean questions (coverage >= 20% or duplicated stem). The 40 dirty questions are a subset of these.
- `summary.csv`, `top_ngrams.txt`: per-benchmark rates and the 25 most frequent shared 13-grams.

## Reproduce the sensitivity results
```bash
python scripts/05_contamination_sensitivity/overlap_sensitivity.py --logs results/lm_eval_logs \
    --flagged results/05_contamination_sensitivity/contamination_out/contaminated_items.json --out sens_dirty
python scripts/05_contamination_sensitivity/overlap_sensitivity.py --logs results/lm_eval_logs \
    --flagged results/05_contamination_sensitivity/contamination_out/not_clean_items.json --out sens_notclean
```
The `all` rows reproduce Table 8 exactly. Expected key values (4-bit MediLITE-QA):

| Subset | n | ECE | AUROC |
|---|---|---|---|
| all | 7,045 | 0.092 | 0.758 |
| excluding 40 dirty | 7,005 | 0.092 | 0.758 |
| excluding 344 not clean | 6,701 | 0.089 | 0.759 |
| MMLU only | 1,089 | 0.048 | 0.823 |

## Screen
```bash
python scripts/05_contamination_sensitivity/contamination_check.py --train training_corpus.jsonl --train-fields text \
    --eval-logs results/lm_eval_logs/medilite --out contamination_out
```
`training_corpus.jsonl` is the corpus file used for the published screen (393,000 records, 8,840,438 words; first line of `top_ngrams.txt`).

## Experiment C sensitivity and selective reasoning
```bash
python scripts/05_contamination_sensitivity/expC_sensitivity.py --csv results/02_answer_modes_energy/energy_items_final.csv \
    --dirty results/05_contamination_sensitivity/contamination_out/contaminated_items.json \
    --notclean results/05_contamination_sensitivity/contamination_out/not_clean_items.json --out expC_sensitivity_out
```
Recomputes Tables 10 and 11 on all 198 questions, without the 1 dirty question and without the 16 not-clean questions,
and reports confidence-gated reasoning, option scoring vs reasoning by confidence quartile, and the oracle upper bound
for selective reasoning. No model is run. The `all` rows reproduce Tables 10 and 11.
