import os, json, csv
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def log_progress(tag, idx, total, steps=20):
    """
    Prints a simple incremental progress message a few times per loop.
    """
    if total <= 0:
        return
    interval = max(1, total // steps)
    if ((idx + 1) % interval == 0) or (idx + 1 == total):
        pct = (idx + 1) / total * 100.0
        print(f"[{tag}] {idx + 1}/{total} ({pct:.1f}%)")

def read_metadata(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        header_skipped = False
        for fn, cn in r:
            if not header_skipped:
                header_skipped = True
                if fn.lower() == "filename" and cn.lower() == "class":
                    continue
            rows.append((fn.strip(), cn.strip()))
    return rows


def paths_from_metadata(metadata, features_dir, rows=None):
    """
    Build the list of feature paths that correspond to metadata entries.
    If rows is provided it must be an iterable of (filename, class) tuples,
    otherwise the metadata file is re-read.
    """
    if rows is None:
        rows = read_metadata(metadata)

    paths = []
    seen = set()
    for fn, _ in rows:
        name = os.path.splitext(fn)[0] + ".npy"
        p = os.path.join(features_dir, name)
        if os.path.exists(p) and p not in seen:
            paths.append(p)
            seen.add(p)

    return paths

def save_metadata(path, items):
    meta = {}
    if os.path.exists(path):
        with open(path, "r", newline='', encoding='utf-8') as f:
            meta = json.load(f)
    meta.update(items)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

# distances

def euclidean_d(X, C):
    """
    X = a feature representation of an audio
    X has the dimension of N x D
        - N the number of feature vectors (frames) in the sound
        - D the number of features in a feature vector

    C = the codebook
    C has the dimension K x D
        - K the number of clusters in the codebook
        - D the number of features

    It returns an array of size N, containing the cluster index for the cluster that is
    closest in euclidian distance to the feature vector
    """
    X2 = np.sum(X * X, axis=1, keepdims=True)
    C2 = np.sum(C*C, axis=1, keepdims=True).T
    D2 = X2 + C2 - 2.0 * (X @ C.T)
    return np.argmin(D2, axis=1)

def cosine_d(X, C):
    """
    Cosine similarity as distance
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Cn.T
    return np.argmax(S, axis=1)

def load_codebook(path):
    z = np.load(path)
    c       = z["codebook"]
    mean    = z["mean"]
    std     = z["std"]
    k       = int(z["k"])
    return c, mean, std, k

def load_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)
