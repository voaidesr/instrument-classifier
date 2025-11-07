PYTHON ?= python
DATA_ROOT ?= data/Raw/Train
TRAIN_METADATA ?= data/Raw/Metadata_Train.csv
TEST_METADATA ?= data/Raw/Metadata_Test.csv
TEST_ROOT ?= data/Raw/Test_submission
FEATURE_DIR ?= data/Processed/Features
HIST_DIR ?= data/Processed/Features_hist
MODELS_DIR ?= models
K ?= 128
MAX_FRAMES ?= 1000
METRIC ?= euclidean
ALPHA ?= 1.0

.PHONY: all pipeline features codebook histograms nb evaluate clean

all: pipeline

pipeline: nb
	@echo "[make] pipeline complete"

features:
	@echo "[make] Extracting MFCC features"
	$(PYTHON) src/extract_features.py --data_root $(DATA_ROOT) --metadata $(TRAIN_METADATA) --out_dir $(FEATURE_DIR)

codebook: features
	@echo "[make] Training vector quantizer (k=$(K))"
	$(PYTHON) src/vq.py --feat_dir $(FEATURE_DIR) --model_dir $(MODELS_DIR) --k $(K) --max_frames_per_file $(MAX_FRAMES) --metadata $(TRAIN_METADATA) --data_root $(DATA_ROOT)

histograms: codebook
	@echo "[make] Building bag-of-audio histograms with $(METRIC) distance"
	$(PYTHON) src/quantize.py --feature_dir $(FEATURE_DIR) --models_dir $(MODELS_DIR) --out_dir $(HIST_DIR) --metric $(METRIC) --metadata $(TRAIN_METADATA) --data_root $(DATA_ROOT)

nb: histograms
	@echo "[make] Training Naive Bayes classifier (alpha=$(ALPHA))"
	$(PYTHON) src/train_nb.py --metadata $(TRAIN_METADATA) --data_root $(DATA_ROOT) --hists_dir $(HIST_DIR) --models_dir $(MODELS_DIR) --alpha $(ALPHA)

evaluate: nb
	@echo "[make] Evaluating on $(TEST_METADATA)"
	$(PYTHON) src/predict.py --metadata $(TEST_METADATA) --data_root $(TEST_ROOT) --models_dir $(MODELS_DIR)

clean:
	@echo "[make] Removing generated artifacts"
	rm -rf $(FEATURE_DIR) $(HIST_DIR) $(MODELS_DIR)/codebook.npz $(MODELS_DIR)/nb_counts.npz
