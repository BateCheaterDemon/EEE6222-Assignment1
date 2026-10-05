# EE6222 Assignment 1 — PCA / LDA Dimensionality Reduction for Face Classification

A mini program for dimensionality reduction and classification on the AT&T (ORL)
face database. It reduces 92×112 face images with **PCA** and **LDA**, classifies
the reduced features with a **minimum Mahalanobis-distance classifier** and a
**nearest-neighbor (1-NN) classifier**, and plots the classification accuracy
against the number of retained dimensions *m* (1 … 199 for PCA, 1 … 39 for LDA).

## Dataset

- AT&T / ORL face database: 40 subjects × 10 grayscale images, 92×112, PGM format
- Expected location in this repository:

```
datasets/att-database-of-faces/
├── s1/   1.pgm … 10.pgm
├── …
└── s40/  1.pgm … 10.pgm
```

- Credit: Olivetti Research Laboratory (Samaria & Harter, 1994).

### Downloading the dataset

If the dataset folder is missing, re-download it from Kaggle with the included
helper script (run it from the repository root, since it writes to a relative path):

```bash
pip install kagglehub        # only needed if not already installed
python datasets_download.py
```

The script fetches `kasikrit/att-database-of-faces` and extracts it into
`datasets/att-database-of-faces/`, recreating the layout shown above.

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy` (core); `matplotlib` (figures)

```bash
conda install numpy scipy matplotlib
# or: pip install numpy scipy matplotlib
```

## Quick start

Run from the repository root:

```bash
python -m src.run_experiment            # baseline preprocessing
python -m src.run_experiment --hist-eq  # with histogram equalization
```

A full run takes well under a minute and writes everything to `outputs/`.

| Flag | Effect |
|---|---|
| `--hist-eq` | Apply per-image histogram equalization before vectorization (off by default) |

## Outputs

```
outputs/
├── figures/
│   ├── montage.png        # sample faces (dataset overview / alignment evidence)
│   ├── mean_eigen.png     # mean face + first 15 eigenfaces
│   └── acc_vs_m.png       # accuracy vs m: PCA (left) and LDA (right), Mahalanobis & 1-NN
└── results/
    ├── curves.csv         # columns: method, m, mahalanobis, 1nn — full accuracy curves
    └── settings.json      # all experiment settings + shuffled-label control accuracy
```

With the default settings (seed 42, histogram equalization on), test accuracy
reaches ≈ 0.95 for PCA and ≈ 0.93 for LDA; the shuffled-label control stays at
≈ 0.03 (chance level 1/40 = 0.025).

## Project layout

| File | Responsibility |
|---|---|
| `src/config.py` | All experiment parameters (seed, split, m ranges, LDA stage, ridge) |
| `src/data_io.py` | PGM parsing and dataset loading |
| `src/preprocess.py` | Scaling to [0,1], histogram equalization, vectorization, centering |
| `src/split.py` | Stratified 5-train / 5-test split per subject |
| `src/dimreduce.py` | PCA (economy SVD) and two-stage LDA |
| `src/classifiers.py` | Minimum Mahalanobis distance and 1-NN |
| `src/evaluate.py` | Accuracy sweep over m |
| `src/visualize.py` | Figures |
| `src/run_experiment.py` | Entry point; runs the whole pipeline |

## Pipeline

1. **Load** all 400 images (92×112) and flatten each to a 10 304-dim vector.
2. **Preprocess**: scale to [0,1]; optionally equalize each image's histogram;
   center with the *training* mean.
3. **Split**: per subject, 5 random images for training and 5 for testing (fixed seed).
4. **PCA**: economy SVD of the centered training matrix; keep the top-m principal
   directions (m = 1 … 199 = N_train − 1).
5. **LDA**: reduce to d = 100 with PCA first, then solve the generalized
   eigenproblem S_B v = λ S_W v and keep the top-m discriminant directions
   (m = 1 … 39 = C − 1).
6. **Project** training and test data with the *same* projection matrix Φ and the
   *training* mean μ: y = Φᵀ(x − μ).
7. **Classify** at every m with minimum Mahalanobis distance (pooled covariance)
   and 1-NN; compute test accuracy.
8. **Plot & save** accuracy vs m and the result files.

## Configuration

All parameters live in `src/config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `SEED` | 42 | Random seed for the train/test split |
| `N_TRAIN_PER_CLASS` | 5 | Training images per subject (the rest are test) |
| `M_PCA_MAX` | 199 | Largest PCA dimension (= N_train − 1) |
| `M_LDA_MAX` | 39 | Largest LDA dimension (= C − 1) |
| `LDA_PCA_STAGE` | 100 | PCA stage dimension used before LDA |
| `RIDGE` | 1e-6 | Scale-relative ridge for covariance regularization |

## Methodology notes

- **Training-only parameters.** μ, Φ, class means and the pooled covariance are
  estimated from training data only; the test set is used solely to score accuracy.
- **Shuffled-label control.** The run prints an accuracy obtained after randomly
  permuting the training labels (expected ≈ 1/40). A much higher value would
  indicate that information is leaking across the split.
- **LDA stage dimension.** `LDA_PCA_STAGE` is 100 rather than N − C = 160: at
  exactly N − C the within-class scatter matrix sits on its rank boundary and the
  leading generalized eigenvectors become numerically unstable.
- **Dimensionality limits.** PCA is capped at N_train − 1 = 199 and LDA at
  C − 1 = 39 discriminant directions.

## References

- F. Samaria and A. Harter, "Parameterisation of a stochastic model for human
  face identification," *2nd IEEE Workshop on Applications of Computer Vision*, 1994.
- X. Jiang, "Linear Subspace Learning-Based Dimensionality Reduction,"
  *IEEE Signal Processing Magazine*, vol. 28, no. 2, pp. 16–26, 2011.
