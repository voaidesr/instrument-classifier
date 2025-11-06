import argparse, os, utils
import numpy as np

def load_codebook(path):
    z = np.load(path)
    c       = z["codebook"]
    mean    = z["mean"]
    std     = z["std"]
    k       = int(z["k"])
    return c, mean, std, k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="data/Processed/Features")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out_dir", default="data/Processed/Features_hist")
    ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    ap.add_argument("--metadata", default="data/Raw/Metadata_Train.csv")
    args = ap.parse_args()

    utils.ensure_dir(args.out_dir)

    path = os.path.join(args.models_dir, "codebook.npz")
    c, mean, std, k = load_codebook(path)
    paths = utils.paths_from_metadata(args.metadata, args.feature_dir)

    for p in paths:
        F = np.load(p).astype(np.float32)
        if F.size == 0:
            continue

        # standardize to match the codebook
        X = (F - mean) / std

        match args.metric:
            case "euclidean":
                idx = utils.euclidean_d(X, c)
            case "cosine":
                idx = utils.cosine_d(X, c)
            case _:
                raise ValueError("Metric unrecognized!")

        h = np.bincount(idx, minlength=k).astype(np.int64)
        base = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(args.out_dir, base + ".npy"), h)

if __name__ == "__main__":
    main()