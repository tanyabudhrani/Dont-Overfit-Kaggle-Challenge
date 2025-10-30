# Don't Overfit! (Kaggle) - Machine Learning Reimplementation

A methodologically sound reimplementation of the classic **“Don’t Overfit!”** Kaggle competition, focusing on *generalization under extreme overfitting risk* (250 samples, 300 features).  

---

## Overview

The goal of this project is to **build a reproducible, regularized machine learning model** capable of generalizing despite a deliberately noisy, small, and high-dimensional dataset.

Unlike leaderboard-exploiting kernels that inferred test correlations, this implementation maintains **methodological integrity** through:
- Strong regularization (L2 penalty)
- Monte Carlo bagging
- Stratified cross-validation
- Noise injection and PCA-based compression

---

## Model Architecture

| Stage | Description |
|-------|--------------|
| **Preprocessing** | Median imputation → Standard scaling → ANOVA F-test (top 25 features) → PCA (8 components) |
| **Classifier** | Logistic Regression (`solver="saga"`, `C=0.0005`, L2 regularization) |
| **Ensemble** | Monte Carlo bagging with 10 randomized seeds and Gaussian noise injection |
| **Validation** | 30-fold Stratified CV using ROC-AUC metric |

---

## Implementation

### Key Code Components
- **Pipeline**: Built with Scikit-learn `Pipeline()` combining imputation, scaling, selection, PCA, and classifier.
- **Noise Injection**: Adds small Gaussian perturbations to input data for robustness.
- **Monte Carlo Bagging**: Averages predictions across multiple seeds for smooth probability estimates.
- **Cross-Validation**: Stratified 30-fold CV to ensure balanced splits and fair AUC evaluation.

---

## Results

| Metric | Value |
|--------|--------|
| **OOF ROC-AUC (CV)** | 0.627 ± 0.015 |
| **Public Leaderboard** | 0.713 |
| **Private Leaderboard** | 0.711 |

## Conclusion

This implementation demonstrates how **regularization, dimensionality reduction, and ensemble averaging** can yield honest, stable results on data designed to trick models into overfitting.  
It emphasizes **methodology over leaderboard performance**, providing a reproducible framework for small-sample, high-dimensional learning.

---

## Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib
````

---

## Usage

1. Place `train.csv` and `test.csv` inside `/mnt/data/`
2. Run:

```bash
python dont_overfit_baseline.py
```

3. Output file:

```
submission.csv
```

---

## References

* Kaggle: [Don’t Overfit! Competition](https://www.kaggle.com/competitions/dont-overfit-ii)
  
* Chris Deotte, *LB Probing Strategies* (2nd Place Kernel)
  
* PolyU DSAI4203 – Machine Learning Course Project

---

**Author:** Tanya Budhrani

**Email:** [tanya.budhrani@connect.polyu.hk](mailto:tanya.budhrani@connect.polyu.hk)

**Date:** October 2025
