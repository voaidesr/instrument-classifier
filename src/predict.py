import argparse, os, utils, csv
import numpy as np
import librosa

def wav_to_histogram(path, mfcc_params, codebook_path):
    codebook, mean, std, k = utils.load_codebook(codebook_path)
    y, sr = librosa.load(path, sr=mfcc_params["sr"], mono=True)
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=mfcc_params["n_mfcc"], n_fft=mfcc_params["n_fft"], hop_length=mfcc_params["hop_length"])

    F = M.T.astype(np.float32)
    X = (F - mean) / (std + 1e-8)
    idx = utils.euclidean_d(X, codebook)
    h = np.bincount(idx, minlength=codebook.shape[0]).astype(np.int64)
    return h

def score_histogram(h, nb_counts_path, alpha, priors):
    z = np.load(nb_counts_path)
    counts = z["counts_by_class"].astype(np.float64)   # (C,k)
    totals = z["class_totals"].astype(np.float64)      # (C,)
    C, k = counts.shape

    denom = totals + alpha * k                         # (C,)
    theta = (counts + alpha) / denom[:, None]          # (C,k)
    log_theta = np.log(theta + 1e-12)
    log_priors = np.log(priors + 1e-12)

    ll = h[None, :] @ log_theta.T                      # (1,C)
    scores = ll.ravel() + log_priors                   # (C,)
    return scores

def _label_names(labels_obj):
    # Accept list OR dict (either name->idx or idx->name, with str/int keys)
    if isinstance(labels_obj, list):
        return labels_obj
    if isinstance(labels_obj, dict):
        items = []
        for k, v in labels_obj.items():
            # try idx->name
            try:
                idx = int(k)
                name = v
            except ValueError:
                # assume name->idx
                name = k
                idx = int(v)
            items.append((idx, str(name)))
        items.sort(key=lambda t: t[0])
        return [name for _, name in items]
    raise TypeError("meta['labels'] must be list or dict")

def _display_label(name):
    prefix = "Sound_"
    if name.startswith(prefix):
        name = name[len(prefix):]
    return name

def _log_scores_to_probs(scores):
    if scores.size == 0:
        return scores
    max_log = float(np.max(scores))
    shifted = np.exp(scores - max_log)
    total = shifted.sum()
    if total <= 0:
        return np.full_like(scores, 1.0 / scores.size)
    return shifted / total

def predict_file(wav_path, models_dir, topk):
    meta = utils.load_meta(os.path.join(models_dir, "meta.json"))
    labels = _label_names(meta["labels"])
    alpha = float(meta.get("nb", {}).get("alpha", 1.0))
    mfcc_params = meta["mfcc_params"]

    h = wav_to_histogram(wav_path, mfcc_params, os.path.join(models_dir, "codebook.npz"))
    z = np.load(os.path.join(models_dir, "nb_counts.npz"))
    priors = z["priors"].astype(np.float64)

    scores = score_histogram(h, os.path.join(models_dir, "nb_counts.npz"), alpha, priors)
    probs = _log_scores_to_probs(scores)
    order = np.argsort(scores)[::-1]
    pred = int(order[0])
    clean_labels = [_display_label(lbl) for lbl in labels]

    green = "\033[32m"
    reset = "\033[0m"
    print("-" * 40)
    print(f"Prediction: {green}{clean_labels[pred]}{reset}")
    print("-" * 40)
    print("Top candidates:")
    for i in order[:topk]:
        i = int(i)
        print(f"  {clean_labels[i]:<10s}  log={scores[i]:8.4f}  prob={probs[i]:.3f}")
    print("-" * 40)


def evaluate_set(metadata_csv, data_root, models_dir, dump_preds=None):
    meta = utils.load_meta(os.path.join(models_dir, "meta.json"))
    labels = meta["labels"]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
    alpha = float(meta.get("nb", {}).get("alpha", 1.0))
    mfcc_params = meta["mfcc_params"]

    # preload nb params
    z = np.load(os.path.join(models_dir, "nb_counts.npz"))
    priors = z["priors"].astype(np.float64)

    rows = utils.read_metadata(metadata_csv)
    C = len(labels)
    conf = np.zeros((C, C), dtype=np.int64)
    n_total = 0
    n_correct = 0

    out_rows = []
    for rel, cls in rows:
        true_ok = cls in label_to_idx
        if not true_ok:
            # skip unknown labels
            continue
        wav_path = os.path.join(data_root, rel)
        if not os.path.exists(wav_path):
            continue

        h = wav_to_histogram(wav_path, mfcc_params, os.path.join(models_dir, "codebook.npz"))
        scores = score_histogram(h, os.path.join(models_dir, "nb_counts.npz"), alpha, priors)
        pred_idx = int(np.argmax(scores))
        true_idx = label_to_idx[cls]

        conf[true_idx, pred_idx] += 1
        n_total += 1
        if pred_idx == true_idx:
            n_correct += 1

        if dump_preds is not None:
            out_rows.append((rel, cls, labels[pred_idx], float(scores[pred_idx])))

    acc = (n_correct / n_total) if n_total else 0.0
    print(f"files: {n_total}")
    print(f"accuracy: {acc:.4f}")
    print("confusion (rows=true, cols=pred):")
    # pretty print
    disp_labels = [_display_label(lbl) for lbl in labels]
    head = " " * 14 + " ".join(f"{lbl:>10s}" for lbl in disp_labels)
    print(head)
    for i, lbl in enumerate(disp_labels):
        row = " ".join(f"{conf[i,j]:10d}" for j in range(C))
        print(f"{lbl:>12s}  {row}")

    # per-class recall and precision
    row_totals = conf.sum(axis=1)
    col_totals = conf.sum(axis=0)
    rec = np.divide(np.diag(conf), row_totals, out=np.zeros(C), where=row_totals!=0)
    prec = np.divide(np.diag(conf), col_totals, out=np.zeros(C), where=col_totals!=0)
    print("recall:")
    for i, lbl in enumerate(disp_labels): print(f"{lbl:>12s}: {rec[i]:.4f}")
    print("precision:")
    for i, lbl in enumerate(disp_labels): print(f"{lbl:>12s}: {prec[i]:.4f}")

    if dump_preds is not None and out_rows:
        os.makedirs(os.path.dirname(dump_preds) or ".", exist_ok=True)
        with open(dump_preds, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "true", "pred", "pred_logscore"])
            w.writerows(out_rows)


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--wav", help="Classify a single WAV")
    mode.add_argument("--metadata", help="CSV with 'filename,class' for evaluation")
    ap.add_argument("--data_root", default="data/Raw/Test_submission", help="Used with --metadata")
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--topk", type=int, default=4, help="Top-k to print in single-file mode")
    ap.add_argument("--dump_preds", default=None, help="Optional CSV to write per-file predictions in eval mode")
    args = ap.parse_args()

    if args.wav:
        predict_file(args.wav, args.models_dir, args.topk)
    else:
        evaluate_set(args.metadata, args.data_root, args.models_dir, dump_preds=args.dump_preds)

if __name__ == "__main__":
    main()
