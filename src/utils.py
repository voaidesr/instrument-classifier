import os, json, csv
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_metadata(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.reader(f)
        cnt = 0
        for fn, cn in r:
            if cnt == 0:
                cnt += 1
                continue
            cnt += 1
            rows.append((fn.strip(), cn.strip()))
    return rows

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
    D2 = X2 + C2 - 2.0 * (X2 @ C2.T)
    return np.argmin(D2, axis=1)

def cosine_d(X, C):
    """
    Cosine similarity as distance
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Cn.T
    return np.argmax(S, axis=1)
