# 00 · Logged evaluation (Kaggle, 2 × NVIDIA T4)

Place `run_all_one_session.py` from the MediLITE-QA study in this folder.

It runs, in one Kaggle session:

1. zero-shot lm-evaluation-harness 0.4.11 in 16-bit for the base model, MediLITE-QA and (with `--with-comparison`) OpenBioLLM-8B, with `--log_samples`;
2. Q4_K_M GGUF scoring of the same logged requests with llama.cpp on the GPU (option log-likelihood, no BOS, context 4,096).

```bash
python run_all_one_session.py --with-comparison
```

The resulting `lm_eval_logs/` folder (one subfolder per model, with `samples_<task>_<timestamp>.jsonl` files) is the input to all other experiments. Its `Scorer` class is also used by `../03_platform_validation/cross_check_scorer.py`.
