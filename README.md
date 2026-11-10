# Confidence and per-question energy of a 4-bit medical language model on a laptop CPU

Code accompanying the manuscript

> Anonymous Authors. *Accuracy, Confidence Calibration, and Energy Consumption of a Quantized Medical Language Model on a Laptop Processor.* Manuscript under review.

The study evaluates **MediLITE-QA**, a Phi-3.5-mini-instruct model fine-tuned for medical question answering and quantized to 4-bit GGUF (Q4_K_M), on a commodity laptop processor. It builds on [Original Author 1] & [Original Author 2] (2026), *Sustainability* 18(21), 11245, https://doi.org/10.3390/su182111245.

---

## Overview

The repository contains the code for four experiments and for the figures, and the per-question results reported in the manuscript (`results/`).

| Experiment | Question | Runs on |
|---|---|---|
| **A. Calibration** | Is the 4-bit fine-tuned model's confidence reliable? (ECE, AUROC, AURC; paired bootstrap) | any CPU |
| **B. Cascade** | Does escalating low-confidence questions to an 8B model save energy or add accuracy? (decision rule fixed in advance) | any CPU |
| **C. Answer modes** | Accuracy and measured energy per question for option scoring, letter generation and step-by-step reasoning | laptop CPU with Intel RAPL |
| **D. Platform validation** | Does laptop (CPU) option scoring reproduce the logged GPU results, and is any difference due to code or platform? | laptop CPU |

Experiments A, B and D reuse per-question logs from the language-model evaluation harness; only Experiment C runs models for energy measurement.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── results/                               # per-question results behind the manuscript (see results/README.md)
└── scripts/
    ├── 00_logged_evaluation/              # Kaggle GPU: fp16 lm-eval + Q4 GGUF scoring (see folder README)
    ├── 01_calibration_cascade/
    │   └── calibration_cascade_pilot.py   # Experiments A and B
    ├── 02_answer_modes_energy/            # Experiment C
    │   ├── export_items.py                # builds the fixed 198-question set
    │   ├── energy_per_item.py             # per-question accuracy, time and RAPL energy
    │   ├── run_energy_day.sh              # runs the full measurement sequence
    │   ├── merge_runs.py                  # combines option-scoring and generation-mode runs
    │   ├── analyze_answer_modes.py        # tables, paired tests, gated-reasoning analysis
    │   ├── rescore_reasoning.py           # first-answer rescoring and runaway counts
    │   └── extract_error_cases.py         # cases for the qualitative error analysis
    ├── 03_platform_validation/            # Experiment D
    │   ├── diagnose_agreement.py
    │   └── cross_check_scorer.py
    └── 04_figures/
        ├── make_figures.py                # Figures 4-6 (PDF and 600 dpi TIFF)
        └── graphical_abstract.py          # graphical abstract (PNG and 500 dpi TIFF)
```

The per-question results reported in the manuscript are stored in `results/`; see [Data](#data). Model files are not included.

---

## Requirements

**Software** (Python 3.10 or later):

```bash
pip install -r requirements.txt
```

`requirements.txt` pins `llama-cpp-python==0.3.35`, the version used for all laptop measurements. Install the CPU build; no GPU is used in Experiments C and D.

The logged evaluation in `00_logged_evaluation/` has its own pinned environment (lm-evaluation-harness 0.4.11 and related packages), installed by that script on Kaggle.

**Hardware used in the paper**

| Item | Setting |
|---|---|
| Laptop | Lenovo ThinkPad E15 Gen 2 |
| Processor | Intel Core i5-1135G7 (4 cores, 8 threads) |
| Memory | 6.96 GiB available |
| OS | Linux, with the Intel RAPL powercap interface |
| Inference | llama-cpp-python 0.3.35, CPU only, 4 threads, context 2,048 |

Energy is read from the RAPL package counter (`/sys/class/powercap/intel-rapl:0/energy_uj`), which on recent kernels needs a one-time permission change after every reboot:

```bash
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj
```

---

## Models

| Role | Model | Hugging Face |
|---|---|---|
| Evaluated model (4-bit) | MediLITE-QA Q4_K_M | `[ANONYMIZED]/MediLITE-QA-GGUF` |
| Evaluated model (16-bit) | Phi-3.5-mini-instruct + MediLITE-QA adapter | `microsoft/Phi-3.5-mini-instruct`, `[ANONYMIZED]/phi3.5-mini-4k-qlora-medical-seg-vall_med` |
| Base model | Phi-3.5-mini-instruct (16-bit and Q4_K_M) | `microsoft/Phi-3.5-mini-instruct`, `QuantFactory/Phi-3.5-mini-instruct-GGUF` |
| Larger medical model | OpenBioLLM-8B (16-bit and Q4_K_M) | `aaditya/Llama3-OpenBioLLM-8B`, `bartowski/OpenBioLLM-Llama3-8B-GGUF` |

Model files are not included; download them from Hugging Face into a local folder (e.g. `../gguf_models/`).

---

## Data

Nine public benchmarks are evaluated zero-shot through lm-evaluation-harness 0.4.11: MedMCQA, MedQA (4 options), PubMedQA and six medical MMLU subsets (anatomy, clinical knowledge, college biology, college medicine, medical genetics, professional medicine); 7,045 questions in total.

The results reported in the manuscript are in [`results/`](results/): the per-question evaluation logs, the 198-question set with its sampling seed, the per-question accuracy, timing and energy records, the run logs with software versions, and the generated texts of both generation runs. `results/README.md` describes every file and maps it to the tables and figures of the manuscript. The benchmark questions contained in these files remain under the licences of MedMCQA, MedQA, PubMedQA and MMLU.

---

## Reproducing the results

Run the steps in order. Paths below assume the commands are run from the repository root (except in step 2), the per-question logs are in `results/lm_eval_logs/` and the GGUF files are in `../gguf_models/`. To check the reported numbers without re-running the laptop measurements, use the archived records in `results/` in place of the files produced in step 2.

### 0. Logged evaluation (Kaggle, 2 × NVIDIA T4)

`scripts/00_logged_evaluation/run_all_one_session.py` produces the fp16 lm-eval logs and the Q4 GGUF scores of the same requests. See the README in that folder.

### 1. Calibration and cascade (Experiments A and B) — any CPU, about 15 min

```bash
python scripts/01_calibration_cascade/calibration_cascade_pilot.py --logs results/lm_eval_logs \
    --models medilite_q4 medilite base openbiollm8b \
    --small medilite_q4 --large openbiollm8b \
    --energy-small 126.237 --energy-large 131.924 \
    --tau-rule maxacc --bootstrap 2000 \
    --pairs medilite:base medilite_q4:medilite medilite:openbiollm8b medilite_q4:openbiollm8b \
    --out pilot_final
```

The two energies are the measured option-scoring energies per question from step 2. Running without `--energy-small/--energy-large` uses the proxy ratio `--energy-ratio 1.68`; the proxy run in the paper used `--bootstrap 300`.

### 2. Answer modes and energy (Experiment C) — laptop CPU

Measurement conditions: mains power, the same power mode throughout, wireless networking off, other applications closed, laptop not in use.

```bash
cd scripts/02_answer_modes_energy

# Fixed question set: 22 per benchmark, seed 1234 (seconds)
python export_items.py --logs ../../results/lm_eval_logs --source medilite --per-task 22 --seed 1234 --out items.jsonl

# Five-question timing and template check (about 5 min)
python energy_per_item.py --model ../../../gguf_models/MediLITE-QA.Q4_K_M.gguf --label timing_test \
    --items items.jsonl --modes loglik letter cot --prompt-format phi3 \
    --limit 5 --idle-secs 5 --cooldown 5 --save-text timing_test_texts.jsonl --out timing_test.csv

# Full measurement sequence (about 3-4 h, unattended)
./run_energy_day.sh --medilite ../../../gguf_models/MediLITE-QA.Q4_K_M.gguf \
                    --openbio  ../../../gguf_models/OpenBioLLM-Llama3-8B-Q4_K_M.gguf \
                    --base     ../../../gguf_models/Phi-3.5-mini-instruct.Q4_K_M.gguf \
                    --items items.jsonl --out energy_items.csv
```

`run_energy_day.sh` measures option scoring for all three models and letter generation and step-by-step reasoning for MediLITE-QA, then runs `analyze_answer_modes.py`. Interrupted runs resume from the last finished question.

If the generation modes are measured separately (as in the paper, see the note below), combine and analyse the runs with:

```bash
python merge_runs.py --old energy_items.csv --new energy_items_phi3.csv --out energy_items_final.csv
python analyze_answer_modes.py --csv energy_items_final.csv --small medilite_q4 --large openbiollm8b_q4 --out modes_final
python extract_error_cases.py --csv energy_items_final.csv --texts texts_medilite_q4_phi3.jsonl \
    --items items.jsonl --model medilite_q4 --out table8_cases
python rescore_reasoning.py --csv energy_items_final.csv --texts texts_medilite_q4_phi3.jsonl \
    --items items.jsonl --model medilite_q4 --out rescored
```

### 3. Platform validation (Experiment D) — laptop CPU

Return to the repository root (`cd ../..`). The question set and option-scoring records from step 2 are in `scripts/02_answer_modes_energy/`.

```bash
python scripts/03_platform_validation/diagnose_agreement.py --logs results/lm_eval_logs \
    --items scripts/02_answer_modes_energy/items.jsonl \
    --csv scripts/02_answer_modes_energy/energy_items.csv --out agreement_out

python scripts/03_platform_validation/cross_check_scorer.py \
    --scorer-script scripts/00_logged_evaluation/run_all_one_session.py \
    --model ../gguf_models/MediLITE-QA.Q4_K_M.gguf --label medilite_q4 --ref medilite_q4 \
    --items scripts/02_answer_modes_energy/items.jsonl \
    --csv scripts/02_answer_modes_energy/energy_items.csv --out crosscheck_medilite_q4.json
```

`diagnose_agreement.py` takes seconds; `cross_check_scorer.py` re-scores the 198 questions with the original GPU scoring code on the CPU (about 15 min).

### 4. Figures

```bash
python scripts/04_figures/make_figures.py        # writes figures/fig4-6 (.pdf, .tiff)
python scripts/04_figures/graphical_abstract.py  # writes graphical_abstract.png/.tiff
```

Values are entered in the scripts from the reports above; update them if your results differ.

---

## Mapping to the manuscript

| Manuscript item | Produced by |
|---|---|
| Table 5, Figure 4, Supplementary Table S3 (per benchmark) | `calibration_cascade_pilot.py` → `REPORT.txt`, `calibration_by_model.csv`, `calibration_by_task.csv`, `paired_differences.csv` |
| Table 6 (cascade) | `calibration_cascade_pilot.py` (proxy run and measured run) |
| Tables 7–8, Figure 5 | `energy_per_item.py`, `merge_runs.py`, `analyze_answer_modes.py` |
| Table 9 (error analysis) | `extract_error_cases.py` (cases), manual categorisation |
| Section 3.5, Supplementary Table S2 (first vs corrected run) | `analyze_answer_modes.py`, `rescore_reasoning.py` |
| Section 3.4, Figure 6 | `diagnose_agreement.py`, `cross_check_scorer.py` |
| Figures 4–6, graphical abstract | `make_figures.py`, `graphical_abstract.py` |

---

## Note on chat templates

The MediLITE-QA GGUF file contains no embedded chat template. When no template is given, llama-cpp-python falls back silently to the Llama-2 format, which differs from the Phi-3.5 format used in fine-tuning. In the first measurement run this caused runaway outputs and distorted accuracy and energy for the generation modes (reported in the manuscript's Section 2.11 and Supplementary Table S2).

All generation-mode results in the paper use the explicit Phi-3.5 template:

```bash
python energy_per_item.py ... --modes letter cot --prompt-format phi3
```

To check the template a GGUF file will use:

```bash
python -c "from llama_cpp import Llama; m=Llama(model_path='MODEL.gguf', vocab_only=True, verbose=False); print(m.chat_format, 'tokenizer.chat_template' in m.metadata)"
```

Option scoring uses the raw evaluation-harness prompt and is not affected.

---

## Citation

If you use this code or the results, please cite the manuscript above (full reference to be added on publication) and the MediLITE-QA paper:

```bibtex
@article{medilite_2026,
  author  = {[Original Author 1] and [Original Author 2]},
  title   = {{MediLITE-QA}: Energy-Efficient Medical Question Answering with a Quantized Domain-Specific Language Model on a Commodity Laptop {CPU}},
  journal = {Sustainability},
  volume  = {18},
  number  = {21},
  pages   = {11245},
  year    = {2026},
  doi     = {10.3390/su182111245}
}
```

See also `CITATION.cff`.

---

## License

Code is released under the MIT License (see `LICENSE`). The benchmarks, including the questions reproduced in `results/`, and the models remain under their own licences.

## Contact

Author information has been temporarily redacted for double-blind peer review.
