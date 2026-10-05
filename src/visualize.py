import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import IMG_H, IMG_W

def save_montage(X_img, path, n=20, cols=10):
    """将图像数据保存为拼接图"""
    fig, axes = plt.subplots(nrows=n // cols, ncols=cols, figsize=(cols, n // cols))
    for ax, img in zip(axes.ravel(), X_img[:n]):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def save_mean_and_eigenfaces(mean_face,comps, path, k=15):
    def norm_img(v):
        v = v.reshape((IMG_H, IMG_W))
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        return v
    ncols = 8
    fig, axes = plt.subplots(2, ncols, figsize=(2 * ncols, 4))
    slots = axes.ravel()  # 2x8=16 个槽位：均值 + 前 k 张 eigenface
    slots[0].imshow(mean_face.reshape((IMG_H, IMG_W)), cmap="gray")
    slots[0].set_title("mean")
    for i in range(k):
        slots[i + 1].imshow(norm_img(comps[i]), cmap="gray")
    for ax in slots:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def save_acc_curves(m_pca, pca_res, m_lda, lda_res, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in [("mahalanobis", "Mahalanobis"), ("1nn", "1-NN")]:
        axes[0].plot(m_pca, pca_res[key], label=label)
        axes[1].plot(m_lda, lda_res[key], label=label)
    axes[0].set(title="PCA", xlabel="m", ylabel="accuracy", ylim=(0, 1.02))
    axes[1].set(title="LDA", xlabel="m", ylim=(0, 1.02))
    for ax in axes: ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)