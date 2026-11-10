# Results

Per-question results behind the manuscript *Accuracy, Confidence Calibration, and Energy Consumption of a Quantized Medical Language Model on a Laptop Processor*. The scripts that produced them are in `../scripts/`; table, figure and section numbers refer to the manuscript, and Tables S1–S4 are in its appendix.

## Contents

| Path | Contents | Manuscript |
|---|---|---|
| `items.jsonl` | The 198-question set used on the laptop: 22 questions per benchmark drawn with seed 1234, with the evaluation-harness prompt, answer options, reference answer and the logged prediction of each model | Section 2.3 |
| `lm_eval_logs/` | Per-question log-likelihoods from the evaluation harness (16-bit models) and from 4-bit GGUF scoring of the same requests (`*_q4`), one folder per model, one file per benchmark. There is no full-benchmark 4-bit log for OpenBioLLM-8B; its accuracies come from the 16-bit logs (Section 2.5) | Experiments A, B, D |
| `01_calibration_cascade/final_run/` | Calibration and cascade with measured energies and 2,000 bootstrap resamples: `REPORT.txt` (summary), `calibration_by_model.csv`, `calibration_by_task.csv`, `paired_differences.csv`, `risk_coverage.csv`, `cascade_curve.csv`, `cascade_summary.json` | Table 5, Fig. 4, Table 6 (measured rows), Table S3 |
| `02_answer_modes_energy/corrected_run_phi3/` | Letter generation and step-by-step reasoning with the Phi-3.5 chat template, merged with the option-scoring rows (`energy_items_final.csv`); summaries in `modes_final/`; error-analysis cases in `table8_cases_phi3.*` (Table 9) | Tables 7–9, Fig. 5 |
| `02_answer_modes_energy/first_run_fallback_template/` | First run, in which the inference library fell back to the Llama-2 chat template; kept for the protocol-deviation analysis and the per-token consistency check. Its `energy_items.csv` also contains the option-scoring rows used in the paper | Sections 2.11, 3.5, 3.6, Table S2 |
| `03_platform_validation/` | Agreement between laptop (CPU) and logged (GPU) option scoring, and the code-versus-platform cross-check | Sections 2.8, 3.4, Fig. 6 |

## Per-question records

`energy_items*.csv` contain one row per model, answer mode, repetition and question:

| Column | Meaning |
|---|---|
| `model`, `mode`, `rep` | Model label, answer mode (`loglik` = option scoring, `letter`, `cot` = step-by-step reasoning), repetition (2 for option scoring and letter generation, 1 for reasoning; the manuscript averages energy over repetitions and takes correctness from the first) |
| `task`, `doc_id`, `gold`, `pred`, `correct` | Benchmark, question identifier, reference option index, predicted option index (−1 if unparsable), correctness |
| `parse_ok`, `conf` | Whether an answer could be parsed; option-scoring confidence (maximum softmax probability) |
| `prompt_tokens`, `gen_tokens` | Prompt tokens and generated tokens |
| `seconds` | Wall-clock time for the question (s) |
| `joules`, `idle_w`, `joules_net` | Gross processor-package energy (J, RAPL), idle package power (W), idle-subtracted energy (J) |
| `finish` | Generation stop reason (`stop` or `length`) |

`*_runs.jsonl` record the settings of each run (model file, threads, context length, library version, prompt format, parsing rule, idle power, start and end times). `texts_*.jsonl` hold the generated text for every letter and reasoning answer; the first-run file is gzip-compressed.

## Notes

- The proxy-ratio rows of Table 6 are not stored here; they are reproduced by running `calibration_cascade_pilot.py` without `--energy-small/--energy-large`, which applies the proxy ratio of 1.68. The manuscript's proxy run used 300 bootstrap resamples (seed 1234), not 2,000.

- Energy is gross CPU-package energy read from the Intel RAPL counter on the laptop described in the manuscript (Table 4).
- The answer modes differ in both prompt format and whether text is generated; comparisons are between inference strategies as implemented.
- The first run's generation rows are superseded by the corrected run and are not used for the main letter and reasoning results (Tables 7–9, Fig. 5); they appear only in Table S2 and the consistency check of Section 3.6.
- Benchmark questions appear in `items.jsonl`, `lm_eval_logs/` and the text files; they remain under the licences of MedMCQA, MedQA, PubMedQA and MMLU.
