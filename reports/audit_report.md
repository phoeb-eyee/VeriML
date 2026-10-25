# VeriML Audit Report

## Baseline

Train on historical data using ALL features (leak included).

- features: income, credit_score, leaky_feature
- train rows: 600, test rows: 200
- accuracy: 0.995
- precision: 0.989
- recall: 1.000
- f1: 0.995
- coefficients:
  - income: +0.193
  - credit_score: +0.177
  - leaky_feature: +4.463

## Ablation

Same data, but the leaky feature is removed.

- features: income, credit_score
- train rows: 600, test rows: 200
- accuracy: 0.905
- precision: 0.863
- recall: 0.946
- f1: 0.903
- coefficients:
  - income: +0.247
  - credit_score: +0.213

## Generalization

Train on historical data, test on new data where the leak is gone.

- features: income, credit_score, leaky_feature
- train rows: 800, test rows: 400
- accuracy: 0.755
- precision: 0.769
- recall: 0.739
- f1: 0.754
- coefficients:
  - income: +0.206
  - credit_score: +0.193
  - leaky_feature: +4.809
