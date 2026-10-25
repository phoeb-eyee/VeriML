# VeriML

VeriML is an AI-powered machine learning auditing tool. It investigates whether
a model learns **meaningful patterns** or leans on **misleading correlations,
data leakage, and patterns that fail to generalize**.

This is the **v0.1 auditing engine**: a command-line tool that demonstrates
*data leakage* on a synthetic dataset using real experiments. The Streamlit
dashboard and LLM explanations come in later versions.

## What it does

It generates a fake "loan approval" dataset with two *genuine* features
(`income`, `credit_score`) and one *planted leak* (`leaky_feature`) that
accidentally encodes the answer for historical records. Then it runs three
experiments:

1. **Baseline** — train with all features (leak included). Accuracy looks great.
2. **Ablation** — remove the leak and retrain. Accuracy drops to the "honest" level,
   and the model's coefficients reveal it was leaning on the leak.
3. **Generalization** — train on historical data, test on *new* data where the leak
   is gone. Accuracy collapses, showing the pattern doesn't transfer.

Every number is computed from a real model fit — nothing is hardcoded.

## Setup

```bash
pip install -r requirements.txt
```

(`pandas` and `numpy` are already installed; `scikit-learn` is the only new
dependency.)

## Run

```bash
python main.py
```

This prints a summary to the terminal and writes:
- `data/dataset.csv` — the generated dataset
- `reports/audit_report.json` — machine-readable results (for the future dashboard)
- `reports/audit_report.md` — human-readable results

## Project layout

```
veriml/
├── data.py    # generates the synthetic dataset (with the planted leak)
└── audit.py   # the three experiments + metric helpers
main.py        # entry point: generate -> run experiments -> print + save report
```
