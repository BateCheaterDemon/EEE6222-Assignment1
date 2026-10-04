import numpy as np
from pathlib import Path
from .config import DATA_DIR,IMG_H, IMG_W

def _read_pgm(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        tokens = []
        while len(tokens) < 4:
            line = f.readline()
            if line.startswith(b"#"):
                continue
            tokens += line.split()
        magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
        assert magic == b"P5"
        data = f.read()
        expected = w * h
        rest = b" ".join(tokens[4:]) if len(tokens) > 4 else b""
        buf = rest + data
        buf = buf[:expected] # 去掉多余的字节
        img = np.frombuffer(buf, dtype=np.uint8)
    assert (h, w) == (IMG_H, IMG_W), (h, w)
    return img.reshape((h, w))

def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    dirs = [p for p in DATA_DIR.iterdir()
        if p.is_dir() and p.name.startswith("s") and p.name[1:].isdigit()]
    dirs.sort(key=lambda p: int(p.name[1:]))          # s1, s2, ..., s40
    assert len(dirs) == 40, f"got {len(dirs)}: {[p.name for p in dirs]}"
    for label, d in enumerate(dirs):
        for i in range(1, 11):  
            img = _read_pgm(d / f"{i}.pgm")
            X.append(img); y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)   