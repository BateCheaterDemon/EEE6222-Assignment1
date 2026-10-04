import numpy as np

def stratified_split(y: np.ndarray, n_train_per_class: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        perm = rng.permutation(idx)
        tr += perm[:n_train_per_class].tolist()
        te += perm[n_train_per_class:].tolist()
    return np.array(sorted(tr)), np.array(sorted(te))

 