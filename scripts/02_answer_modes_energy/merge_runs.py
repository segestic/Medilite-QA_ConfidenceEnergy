"""
merge_runs.py
=============
Builds the final per-question table for Experiment C:
  * option-scoring rows (all models) from the original run, and
  * letter and reasoning rows from the rerun with the Phi-3.5 template.
The original letter/reasoning rows (llama-2 fallback template) are left out of
the final table but stay in the original file for the record.

Usage:
  python merge_runs.py --old energy_items.csv --new energy_items_phi3.csv --out energy_items_final.csv
"""
import argparse, csv

ap = argparse.ArgumentParser()
ap.add_argument("--old", default="energy_items.csv")
ap.add_argument("--new", default="energy_items_phi3.csv")
ap.add_argument("--out", default="energy_items_final.csv")
a = ap.parse_args()

old = [r for r in csv.DictReader(open(a.old)) if r["mode"] == "loglik"]
new = [r for r in csv.DictReader(open(a.new)) if r["mode"] in ("letter", "cot")]
fields = list(csv.DictReader(open(a.old)).fieldnames)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(old + new)
count = lambda rows: sorted({(r["model"], r["mode"]) for r in rows})
print(f"option scoring rows from {a.old}: {len(old)} {count(old)}")
print(f"letter/reasoning rows from {a.new}: {len(new)} {count(new)}")
print(f"written {a.out}")
