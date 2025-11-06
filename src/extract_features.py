import utils, argparse, os
import librosa
import numpy as np

"""
Process the raw data (.wav) files by extracting sound features (MFCC).
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/Raw/Train/")
    ap.add_argument("--metadata", default="./data/Raw/Metadata_Train.csv")
    ap.add_argument("--out_dir", default="data/Processed/")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--max_duration", type=float, default=0.0, help="seconds; 0=all")
    args = ap.parse_args()

    utils.ensure_dir(args.out_dir)
    rows = utils.read_metadata(args.metadata)

    mfcc_params = dict(
        sr = args.sr,
        n_mfcc = args.n_mfcc,
        n_fft= args.n_fft,
        hop_length = args.hop_length,
    )


    for fn, _ in rows:
        wav_path = os.path.join(args.data_root, fn)
        y, sr = librosa.load(wav_path, sr=args.sr, mono=True)

        if args.max_duration > 0:
            y = y[:int(args.max_duration * sr)]

        M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length)

        feats = M.T.astype(np.float32) # TODO: see if float32 and float64 makes any difference

        name = fn[:-4]
        out_path = os.path.join(args.out_dir + name + ".npy")
        np.save(out_path, feats)

    utils.save_metadata("models/meta.json", {"mfcc_params": mfcc_params})
if __name__ == '__main__':
    main()