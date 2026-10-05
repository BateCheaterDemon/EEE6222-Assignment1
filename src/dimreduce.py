import numpy as np
from scipy.linalg import eigh

class PCA:
    def __init__(self, m_max: int):
        self.m_max = m_max
        self.mean_ = None
        self.components_ = None
        self.singular_ = None

    def fit(self, Xc: np.ndarray):
        """训练 PCA 模型"""
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

        self.components_ = Vt[:self.m_max]
        self.singular_ = s[:self.m_max]
        self.eigvals_ = (s[:self.m_max] ** 2) / (Xc.shape[0] - 1)
        self.explain_var_ratio_ = self.eigvals_ / self.eigvals_.sum()
        return self

    def transform(self, X: np.ndarray, m: int) -> np.ndarray:
        return X @ self.components_[:m].T


class LDA:
    def __init__(self, pca: PCA, d_stage: int, m_max: int, ridge: float = 1e-6):
        self.pca = pca
        self.d_stage = d_stage
        self.m_max = m_max
        self.ridge = ridge
        self.W_ = None
        self.eigvals_ = None

    def fit(self, Xc: np.ndarray, y_tr: np.ndarray):
        """训练 LDA 模型"""
        Z = self.pca.transform(Xc, self.d_stage) # (N, d) 
        N, d = Z.shape
        classes = np.unique(y_tr)
        mu = Z.mean(axis=0)

        S_W = np.zeros((d, d))
        S_B = np.zeros((d, d))
        for c in classes:
            Zc = Z[y_tr == c]
            mu_c = Zc.mean(axis=0)
            diff = Zc - mu_c
            S_W += diff.T @ diff
            diff_mu = (mu_c - mu).reshape(-1, 1)
            S_B += Zc.shape[0] * (diff_mu @ diff_mu.T)  

        S_W += self.ridge * np.trace(S_W) * np.eye(d)  # 尺度化正则（与分类器一致）

        vals, vecs = eigh(S_B, S_W)
        vals, vecs = vals[::-1], vecs[:, ::-1]  # 按特征值从大到小排序
        vals = np.real(vals)
        vecs = np.real(vecs)

        self.eigvals_ = vals[:self.m_max]
        self.W_ = vecs[:, :self.m_max]
        return self

    def transform(self, Xc: np.ndarray, m: int) -> np.ndarray:
        Z = self.pca.transform(Xc, self.d_stage)
        return Z @ self.W_[:, :m]

            
                              

    
