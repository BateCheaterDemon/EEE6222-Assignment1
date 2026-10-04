import numpy as np
from scipy.linalg import pinv

class Mahalanobis:
    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge
        self.means_ = None
        self.Sigma_inv_ = None
        self.classes_ = None

    def fit(self, Z: np.ndarray, y: np.ndarray):
        """训练 Mahalanobis 模型"""
        self.classes_ = np.unique(y)
        d = Z.shape[1]
        means, S_W = [], np.zeros((d, d))
        for c in self.classes_:
            Zc = Z[y == c]
            mu_c = Zc.mean(axis=0)
            means.append(mu_c)
            diff = Zc - mu_c
            S_W += diff.T @ diff

        self.means_ = np.stack(means)
        S_W /= max(len(Z) - len(self.classes_), 1)
        S_W += self.ridge * np.trace(S_W) * np.eye(d)  # 正则化
        self.Sigma_inv_ = pinv(S_W)
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        """预测"""
        out = np.empty(len(Z), dtype=np.int64)
        for i, z in enumerate(Z):
            diff = self.means_ - z
            d2 = np.einsum("cd,de,ce->c", diff, self.Sigma_inv_, diff)
            out[i] = self.classes_[np.argmin(d2)]
        return out


class OneNN:
    def __init__(self):
        self.gallery_ = None
        self.labels_ = None

    def fit(self, Z: np.array, y: np.ndarray):
        self.gallery_ = Z
        self.labels_ = y
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        d2 = ((Z[:, None, :] - self.gallery_[None, :, :]) ** 2).sum(-1)
        return self.labels_[np.argmin(d2, axis=1)]


    