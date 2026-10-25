"""VeriML v0.1 — run the auditing engine end-to-end.

Usage:
    python main.py
"""

import json
from pathlib import Path

from veriml.audit import run_ablation, run_baseline, run_generalization
from veriml.data import generate_dataset

DATA_PATH = "data/dataset.csv"
REPORT_DIR = Path("reports")


def _print_experiment(exp: dict) -> None:
    print(f"\n=== {exp['name'].upper()} ===")
    print(f"    {exp['description']}")
    print(f"    features used: {exp['features']}")
    print(f"    train rows: {exp['train_rows']}, test rows: {exp['test_rows']}")
    print(f"    accuracy : {exp['accuracy']:.3f}")
    print(f"    precision: {exp['precision']:.3f}")
    print(f"    recall   : {exp['recall']:.3f}")
    print(f"    f1       : {exp['f1']:.3f}")
    if "coefficients" in exp:
        print("    coefficients:")
        for feature, coef in exp["coefficients"].items():
            print(f"        {feature:<14} {coef:+.3f}")


def _write_markdown(experiments: list, path: Path) -> None:
    lines = ["# VeriML Audit Report\n"]
    for exp in experiments:
        lines.append(f"## {exp['name'].capitalize()}\n")
        lines.append(f"{exp['description']}\n")
        lines.append(f"- features: {', '.join(exp['features'])}")
        lines.append(f"- train rows: {exp['train_rows']}, test rows: {exp['test_rows']}")
        lines.append(f"- accuracy: {exp['accuracy']:.3f}")
        lines.append(f"- precision: {exp['precision']:.3f}")
        lines.append(f"- recall: {exp['recall']:.3f}")
        lines.append(f"- f1: {exp['f1']:.3f}")
        if "coefficients" in exp:
            lines.append("- coefficients:")
            for feature, coef in exp["coefficients"].items():
                lines.append(f"  - {feature}: {coef:+.3f}")
        lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    print("Generating the synthetic dataset...")
    df = generate_dataset(path=DATA_PATH)
    print(f"Saved {len(df)} rows to {DATA_PATH}")

    print("\nRunning experiments...")
    experiments = [
        run_baseline(df),
        run_ablation(df),
        run_generalization(df),
    ]

    for exp in experiments:
        _print_experiment(exp)

    # Save a machine-readable report for the future dashboard.
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "audit_report.json"
    json_path.write_text(json.dumps(experiments, indent=2))
    print(f"\nSaved JSON report to {json_path}")

    # Also write a human-readable markdown report.
    md_path = REPORT_DIR / "audit_report.md"
    _write_markdown(experiments, md_path)
    print(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()
