import numpy as np
from .dimreduce import PCA, LDA
from .classifiers import Mahalanobis, OneNN

def sweep(pca: PCA, lda: LDA, Xc_tr, y_tr, Xc_te, y_te, m_list, method: str):
    """返回 {'mahalanobis': [...], '1nn': [...]}，长度 = len(m_list)。"""
    acc_maha, acc_1nn = [], []
    for m in m_list:
        if method == "pca":
            Z_tr = pca.transform(Xc_tr, m)
            Z_te = pca.transform(Xc_te, m)
        else:
            Z_tr = lda.transform(Xc_tr, m)
            Z_te = lda.transform(Xc_te, m)

        clf = Mahalanobis().fit(Z_tr, y_tr)
        acc_maha.append((clf.predict(Z_te) == y_te).mean())

        clf = OneNN().fit(Z_tr, y_tr)
        acc_1nn.append((clf.predict(Z_te) == y_te).mean())
    return {"mahalanobis": acc_maha, "1nn": acc_1nn}

