#!/usr/bin/env bash
# Energy-per-item day on the MediLITE-QA paper laptop (CPU only).
# No file names are built in: pass the GGUF files and (optionally) the item/output files.
#
# Usage:
#   ./run_energy_day.sh --medilite PATH.gguf --openbio PATH.gguf [--base PATH.gguf]
#                       [--items items.jsonl] [--out energy_items.csv] [--threads 4] [--n-ctx 2048]
#
# Labels written to the CSV (and compared with the lm-eval log folders of the same
# names): medilite_q4, openbiollm8b_q4 (compared with fp16 'openbiollm8b'), base_q4.
#
# Before starting: plug in the charger, use the same power mode as in the paper,
# close other applications, and disable Wi-Fi.
set -euo pipefail

MEDILITE="" OPENBIO="" BASE="" ITEMS="items.jsonl" OUT="energy_items.csv" THREADS=4 NCTX=2048
while [ $# -gt 0 ]; do
  case "$1" in
    --medilite) MEDILITE="$2"; shift 2 ;;
    --openbio)  OPENBIO="$2";  shift 2 ;;
    --base)     BASE="$2";     shift 2 ;;
    --items)    ITEMS="$2";    shift 2 ;;
    --out)      OUT="$2";      shift 2 ;;
    --threads)  THREADS="$2";  shift 2 ;;
    --n-ctx)    NCTX="$2";     shift 2 ;;
    -h|--help)  sed -n 2,13p "$0"; exit 0 ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
[ -n "$MEDILITE" ] && [ -n "$OPENBIO" ] || { echo "--medilite and --openbio are required (see --help)"; exit 1; }

# resolve paths before changing directory
abs() { realpath -e "$1" 2>/dev/null || { echo "missing file: $1" >&2; exit 1; }; }
MEDILITE=$(abs "$MEDILITE"); OPENBIO=$(abs "$OPENBIO"); ITEMS=$(abs "$ITEMS")
[ -n "$BASE" ] && BASE=$(abs "$BASE")
OUT=$(realpath -m "$OUT")
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$OUT")"                          # all outputs land next to --out
exec > >(tee -a energy_day.log) 2>&1            # keep a copy of all console output

python -c "import llama_cpp, numpy, scipy; print('llama-cpp-python', llama_cpp.__version__)"
sudo chmod o+r /sys/class/powercap/intel-rapl:0/energy_uj   # RAPL readable without root (resets at reboot)
COMMON=(--items "$ITEMS" --threads "$THREADS" --n-ctx "$NCTX" --out "$OUT")

# Steps 1-3: MediLITE-QA Q4, all three answer modes
# (--prompt-format phi3: the GGUF file has no embedded chat template; see README)
python "$SCRIPTS/energy_per_item.py" --model "$MEDILITE" --label medilite_q4 --modes loglik letter cot \
    --prompt-format phi3 --ref medilite_q4 --save-text texts_medilite_q4.jsonl "${COMMON[@]}"
sleep 60

# Step 4: OpenBioLLM-8B Q4, option log-likelihood only (compared with the fp16 logs)
python "$SCRIPTS/energy_per_item.py" --model "$OPENBIO" --label openbiollm8b_q4 --modes loglik \
    --ref openbiollm8b "${COMMON[@]}"

# Step 5 (optional): base model control
if [ -n "$BASE" ]; then
  sleep 60
  python "$SCRIPTS/energy_per_item.py" --model "$BASE" --label base_q4 --modes loglik --ref base_q4 "${COMMON[@]}"
fi

python "$SCRIPTS/analyze_answer_modes.py" --csv "$OUT" --small medilite_q4 --large openbiollm8b_q4 \
    --out "$(dirname "$OUT")/modes_out"
