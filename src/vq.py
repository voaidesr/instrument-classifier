import argparse, os, glob, utils
import numpy as np
from sklearn.cluster import KMeans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", default="data/Processed")
    ap.add_argument("--model_dir", default="models")
    ap.add_argument("--k", type=int, default="128")
    ap.add_argument("--max_frames_per_file", type=int, default=1000)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--metadata", default="data/Raw/Metadata_Train.csv")
    args = ap.parse_args()

    utils.ensure_dir(args.model_dir)
    paths = utils.paths_from_metadata(args.metadata, args.feat_dir)

    Xs = []
    for p in paths:
        F = np.load(p)

        if F.size == 0: continue
        if F.shape[0] > args.max_frames_per_file:
            idx = np.random.default_rng().choice(F.shape[0], args.max_frames_per_file, replace=False)
            F = F[idx]
        Xs.append(F)

    X = np.vstack(Xs).astype(np.float32)
    d = X.shape[1]

    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8

    # standardize (center at zero and scale to unit variance)
    Xz = (X - mean) / std

    km = KMeans(n_clusters=args.k, n_init=10, max_iter=300, random_state=args.random_state, verbose=0)
    km.fit(Xz)

    codebook = km.cluster_centers_.astype(np.float32)

    path = os.path.join(args.model_dir, "codebook.npz")
    np.savez(path, codebook=codebook, k=np.int64(args.k), d=np.int64(d), mean=mean.astype(np.float32), std=std.astype(np.float32))

    path = os.path.join(args.model_dir, 'meta.json')
    utils.save_metadata(path,
        {
            "vq": {
                "k": args.k,
                "d": int(d),
                "random_state": args.random_state
            }
        })

if __name__ == "__main__":
    main()