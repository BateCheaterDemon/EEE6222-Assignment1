from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "datasets" / "att-database-of-faces"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"

SEED = 42
N_TRAIN_PER_CLASS = 5 #每人5张训练，5张测试
IMG_H, IMG_W = 112, 92
D = IMG_H * IMG_W
N_CLASSES = 40              # C
M_PCA_MAX = 199             # N_tr（总共200样本） - 1
M_LDA_MAX = 39              # C - 1
LDA_PCA_STAGE = 100         # 160=N_tr−C 处 S_W 恰在秩边界、数值奇异；取 100 留余量
RIDGE = 1e-6                # 正则项，用于数值稳定