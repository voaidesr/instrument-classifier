import argparse, os, utils
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/Raw/Metadata_Train.csv")
    ap.add_argument("--data_root", default="data/Raw/Train")
    ap.add_argument("--hists_dir", default="data/Processed/Features_hist")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--alpha", type=float, default=5.0, help="Laplace smoothing to store for inference")
    args = ap.parse_args()

    utils.ensure_dir(args.models_dir)
    rows = utils.read_training_metadata(args.metadata, args.data_root)
    print(f"[nb] preparing label map from {len(rows)} metadata rows")

    label_names = []
    label_to_idx = {}
    for _, cls in rows:
        if cls not in label_to_idx:
            label_to_idx[cls] = len(label_names)
            label_names.append(cls)

    examples = []
    for rel, cls in rows:
        base = os.path.splitext(rel)[0] + ".npy"
        path = os.path.join(args.hists_dir, base)
        if os.path.exists(path):
            examples.append((path, cls))

    if not examples:
        raise RuntimeError(f"No histograms found in {args.hists_dir} matching {args.metadata}")
    print(f"[nb] accumulating counts from {len(examples)} histograms in {args.hists_dir}")

    k = int(np.load(examples[0][0]).shape[0])
    counts_by_class = np.zeros((len(label_names), k), dtype=np.int64)
    class_counts = np.zeros(len(label_names), dtype=np.int64)

    for ex_idx, (path, cls) in enumerate(examples):
        h = np.load(path).astype(np.int64)
        if h.shape[0] != k:
            raise ValueError(f"Histogram at {path} has length {h.shape[0]}, expected {k}")
        lbl_idx = label_to_idx[cls]
        counts_by_class[lbl_idx] += h
        class_counts[lbl_idx] += 1
        utils.log_progress("nb:counts", ex_idx, len(examples))

    total_examples = int(class_counts.sum())
    if total_examples == 0:
        raise RuntimeError("No training examples were counted for Naive Bayes training")

    priors = class_counts.astype(np.float64) / float(total_examples)
    class_totals = counts_by_class.sum(axis=1)

    path = os.path.join(args.models_dir, "nb_counts.npz")
    np.savez(path, counts_by_class=counts_by_class, class_totals=class_totals, priors=priors.astype(np.float64))

    path = os.path.join(args.models_dir, "meta.json")
    utils.save_metadata(path, {"labels": label_names, "nb": {"alpha": args.alpha}})

if __name__ == "__main__":
    main()
