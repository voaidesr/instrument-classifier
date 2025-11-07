import argparse, os, utils
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_dir", default="data/Processed/Features")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--out_dir", default="data/Processed/Features_hist")
    ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    ap.add_argument("--metadata", default="data/Raw/Metadata_Train.csv")
    ap.add_argument("--data_root", default="data/Raw/Train")
    args = ap.parse_args()

    utils.ensure_dir(args.out_dir)

    path = os.path.join(args.models_dir, "codebook.npz")
    c, mean, std, k = utils.load_codebook(path)
    rows = utils.read_training_metadata(args.metadata, args.data_root)
    paths = utils.paths_from_metadata(args.metadata, args.feature_dir, rows=rows)
    total = len(paths)
    print(f"[quantize] converting {total} feature files into histograms using {args.metric} distance")

    for path_idx, p in enumerate(paths):
        F = np.load(p).astype(np.float32)
        if F.size == 0:
            continue

        # standardize to match the codebook
        X = (F - mean) / std

        match args.metric:
            case "euclidean":
                assignments = utils.euclidean_d(X, c)
            case "cosine":
                assignments = utils.cosine_d(X, c)
            case _:
                raise ValueError("Metric unrecognized!")

        h = np.bincount(assignments, minlength=k).astype(np.int64)
        base = os.path.splitext(os.path.basename(p))[0]
        np.save(os.path.join(args.out_dir, base + ".npy"), h)
        utils.log_progress("quantize", path_idx, total)

if __name__ == "__main__":
    main()
