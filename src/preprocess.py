import numpy as np

def histogram_equalize(img: np.ndarray) -> np.ndarray:
    """直方图均衡化"""
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # 归一化
    lut = np.round(cdf * 255).astype(np.uint8)
    return lut[u8].astype(np.float32) / 255.0

def build_X(X_img: np.ndarray, use_hist_eq: bool = False) -> np.ndarray:
    """将图像数据转换为特征向量"""
    X = X_img.astype(np.float64) / 255.0
    if use_hist_eq:
        X = np.stack([histogram_equalize(x) for x in X]) 
    return X.reshape((len(X), -1))

def center_train(X_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只用训练集算 μ。返回 (mu, Xc)。"""
    mu = X_tr.mean(axis=0, keepdims=True)
    return mu, X_tr - mu