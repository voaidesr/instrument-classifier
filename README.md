# Instrument Classification using the Multinomial Naive Bayes Method

>[!warning]
> Still in development.

## Overview

This repository implements a lightweight bag-of-audio pipeline for identifying musical instruments. Raw WAV clips are converted to MFCC frames, vector-quantized into a codebook (k-means), collapsed into bag-of-audio histograms, and finally scored with a Laplace-smoothed multinomial Naive Bayes classifier. The defaults target the provided metadata files (`Metadata_Train.csv`, `Metadata_Test.csv`) whose rows follow `filename,class`.

## Installation

1. Create and activate a virtual environment (Python 3.10+ works; examples below use 3.13).
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Install runtime dependencies.
   ```bash
   python -m pip install -r requirements.txt
   ```
   The pipeline relies on `librosa`, `numpy`, `pandas`, and `scikit-learn`. Librosa may require system packages such as `libsndfile` or `ffmpeg`, depending on your OS.

## Data Layout

The scripts assume the following structure (feel free to adjust paths via CLI flags or Make variables):

```
data/
  Raw/
    Train/                # WAV files referenced by Metadata_Train.csv
    Test_submission/      # WAV files referenced by Metadata_Test.csv (or a submission set)
    Metadata_Train.csv    # filename,class header
    Metadata_Test.csv
  Processed/
    Features/             # MFCC numpy arrays (*.npy)
    Features_hist/        # Bag-of-audio histograms
models/
  meta.json               # Aggregated hyperparameters
  codebook.npz            # K-means codebook + normalization stats
  nb_counts.npz           # Trained NB counts and priors
```

Example metadata row:

```
filename,class
Sound_Guitar_0001.wav,Sound_Guitar
```

## Training Pipeline

The easiest path is to let `make` orchestrate every stage (defaults mirror the paths shown above):

```bash
make pipeline                      # runs features -> codebook -> histograms -> nb
make evaluate                      # trains if needed, then scores Metadata_Test.csv
```

To run the steps manually:

1. **Extract MFCC features**
   ```bash
   python src/extract_features.py --data_root data/Raw/Train --metadata data/Raw/Metadata_Train.csv --out_dir data/Processed/Features
   ```
2. **Fit the vector-quantizer (codebook)**
   ```bash
   python src/vq.py --feat_dir data/Processed/Features --model_dir models --k 128 --max_frames_per_file 1000
   ```
3. **Convert features to bag-of-audio histograms**
   ```bash
   python src/quantize.py --feature_dir data/Processed/Features --models_dir models --out_dir data/Processed/Features_hist --metric euclidean
   ```
4. **Train the multinomial Naive Bayes classifier**
   ```bash
   python src/train_nb.py --metadata data/Raw/Metadata_Train.csv --hists_dir data/Processed/Features_hist --models_dir models --alpha 5.0
   ```

Each script exposes additional CLI arguments (`-h`) for fine-tuning frame limits, distance metrics, alpha, and file locations.

## Using the Models

- **Classify a single WAV**
  ```bash
  python src/predict.py --wav path/to/file.wav --models_dir models --topk 4
  ```
  The script prints the predicted instrument plus the top-k candidates with log-scores and normalized probabilities.

- **Evaluate a labeled set**
  ```bash
  python src/predict.py --metadata data/Raw/Metadata_Test.csv --data_root data/Raw/Test_submission --models_dir models --dump_preds reports/test_preds.csv
  ```
  In evaluation mode you receive accuracy, confusion matrix, per-class precision/recall, and (optionally) a CSV of per-file predictions.

## Outputs and Metadata

- `data/Processed/Features/*.npy`: MFCC frames per recording, saved as `(num_frames, n_mfcc)`.
- `data/Processed/Features_hist/*.npy`: Histogram counts of codebook assignments for each file.
- `models/codebook.npz`: Contains the codebook, feature dimension, and normalization statistics (`mean`, `std`, `k`).
- `models/nb_counts.npz`: Stores per-class counts, overall counts, and learned priors for inference.
- `models/meta.json`: Consolidates MFCC parameters, VQ settings, label names, and Naive Bayes hyperparameters; the prediction script reads this file to stay in sync with the training configuration.

## Results

Using the Euclidean Metric, with the preset values:

```
accuracy: 0.8750
confusion (rows=true, cols=pred):
                  Guitar       Drum      Piano     Violin
      Guitar          17          0          3          0
        Drum           1         19          0          0
       Piano           2          1         16          1
      Violin           2          0          0         18
```
