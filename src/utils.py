import os, json, csv, warnings
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


def read_training_metadata(path, data_root):
    """
    Returns list of (filename, class) pairs for training.
    The provided metadata file in this dataset ships with every Sound_Violin entry
    pointing to the same Drum recordings, while the real violin audio files are
    free-floating in the data_root directory. We detect these conflicting labels
    and swap in the unreferenced violin recordings instead so that every class
    has unique training examples.
    """
    rows = read_metadata(path)
    rows_clean = []
    seen = {}
    conflicts = []
    for fn, cls in rows:
        prev = seen.get(fn)
        if prev is None:
            seen[fn] = cls
            rows_clean.append((fn, cls))
        elif prev == cls:
            continue
        else:
            conflicts.append((fn, prev, cls))

    if not conflicts:
        return rows_clean

    violin_conflicts = [
        fn for fn, a, b in conflicts
        if {"Sound_Violin", "Sound_Drum"} == {a, b}
    ]
    other_conflicts = [c for c in conflicts if c[0] not in violin_conflicts]
    if other_conflicts:
        raise RuntimeError(f"Unexpected conflicting labels in metadata: {other_conflicts[:5]}")

    wave_files = sorted(
        fn for fn in os.listdir(data_root)
        if fn.lower().endswith(".wav")
    )
    unused = [fn for fn in wave_files if fn not in seen]
    if len(unused) < len(violin_conflicts):
        raise RuntimeError(
            f"Need {len(violin_conflicts)} unused .wav files to repair violin metadata, "
            f"but only found {len(unused)} in {data_root}"
        )

    warnings.warn(
        f"Metadata contained {len(violin_conflicts)} Sound_Violin entries that reused Drum files. "
        "Reassigning the unreferenced violin recordings from disk to fix training labels.",
        RuntimeWarning,
        stacklevel=2,
    )

    for fn in unused[:len(violin_conflicts)]:
        seen[fn] = "Sound_Violin"
        rows_clean.append((fn, "Sound_Violin"))

    return rows_clean

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
