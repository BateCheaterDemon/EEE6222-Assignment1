import argparse, json, csv
import numpy as np
from pathlib import Path

from . import config as C
from .data_io import load_dataset
from .preprocess import build_X, center_train
from .split import stratified_split
from .dimreduce import PCA, LDA
from .evaluate import sweep
from .visualize import save_montage, save_mean_and_eigenfaces, save_acc_curves

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hist-eq", action="store_true", help="是否使用直方图均衡化")
    args = ap.parse_args()

    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.RES_DIR.mkdir(parents=True, exist_ok=True)

    X_img, y = load_dataset()
    print("X_img.shape:", X_img.shape, "y.shape:", y.shape)
    X = build_X(X_img, use_hist_eq=args.hist_eq)
    tr, te = stratified_split(y, C.N_TRAIN_PER_CLASS, C.SEED)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == 400
    print(f"split: train={len(tr)}, test={len(te)}")

    mu, Xc = center_train(X[tr])
    Xc_tr, Xc_te = X[tr] - mu, X[te] - mu
    y_tr, y_te = y[tr], y[te]

    # PCA
    pca = PCA(m_max=C.M_PCA_MAX)
    pca.fit(Xc_tr)

    # LDA
    lda = LDA(pca, d_stage=C.LDA_PCA_STAGE, m_max=C.M_LDA_MAX, ridge=C.RIDGE)
    lda.fit(Xc_tr, y_tr)
    # 自检：LDA 特征值只有 39 个显著
    print("LDA top-5 eigvals:", lda.eigvals_[:5])
    print("LDA 40th eigval (should be ~0):", lda.eigvals_[38] if len(lda.eigvals_) > 38 else "N/A")

    
    # --- m 扫描 ---
    m_pca = list(range(1, C.M_PCA_MAX + 1))
    m_lda = list(range(1, C.M_LDA_MAX + 1))
    pca_res = sweep(pca, lda, Xc_tr, y_tr, Xc_te, y_te, m_pca, "pca")
    lda_res = sweep(pca, lda, Xc_tr, y_tr, Xc_te, y_te, m_lda, "lda")

    # --- 存结果 ---
    with open(C.RES_DIR / "curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "m", "mahalanobis", "1nn"])
        for m, a, b in zip(m_pca, pca_res["mahalanobis"], pca_res["1nn"]):
            w.writerow(["pca", m, f"{a:.4f}", f"{b:.4f}"])
        for m, a, b in zip(m_lda, lda_res["mahalanobis"], lda_res["1nn"]):
            w.writerow(["lda", m, f"{a:.4f}", f"{b:.4f}"])

    # --- 图 ---
    save_montage(X_img[tr], C.FIG_DIR / "montage.png")
    save_mean_and_eigenfaces(mu, pca.components_, C.FIG_DIR / "mean_eigen.png")
    save_acc_curves(m_pca, pca_res, m_lda, lda_res, C.FIG_DIR / "acc_vs_m.png")

    # --- 打乱标签对照（无泄漏证据）---
    rng = np.random.default_rng(0)
    y_shuf = rng.permutation(y_tr)
    lda_shuf = LDA(pca, C.LDA_PCA_STAGE, C.M_LDA_MAX, C.RIDGE).fit(Xc_tr, y_shuf)
    from .classifiers import Mahalanobis
    Z_tr = lda_shuf.transform(Xc_tr, 20); Z_te = lda_shuf.transform(Xc_te, 20)
    acc_shuf = (Mahalanobis().fit(Z_tr, y_shuf).predict(Z_te) == y_te).mean()
    print(f"shuffled-label accuracy (expect ~0.025): {acc_shuf:.4f}")

    # --- 设置表 ---
    settings = dict(seed=C.SEED, n_train_per_class=C.N_TRAIN_PER_CLASS,
                    hist_eq=args.hist_eq, m_pca_max=C.M_PCA_MAX,
                    m_lda_max=C.M_LDA_MAX, lda_pca_stage=C.LDA_PCA_STAGE,
                    ridge=C.RIDGE)
    (C.RES_DIR / "settings.json").write_text(json.dumps(settings, indent=2))
    print("done.")

if __name__ == "__main__":
    main()