"""
=========================================================
Project Configuration File
=========================================================

This file contains all configurable parameters used
throughout the project.

=========================================================
"""

from pathlib import Path


# =========================================================
# Project Paths
# =========================================================

# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "cleaned_data"
SPLITED_DATA = DATA_DIR / "splits" 

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Results
RESULTS_DIR = PROJECT_ROOT / "results"

FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
COPARISON_DIR = RESULTS_DIR / "comparison"

# Stopwords
STOPWORDS_PATH = DATA_DIR / "stopwords.txt"


# =========================================================
# Spilit data
# =========================================================

TEST_SIZE = 0.2  # نسبت داده‌های Train

RANDOM_SEED_SPLIT = 42

TEST_DATA_PATH = SPLITED_DATA / "test.csv"

TRAIN_DATA_PATH = SPLITED_DATA / "train.csv"

SPLIT_INFO = SPLITED_DATA / "split_info.json"

# =========================================================
# Dataset Configuration
# =========================================================

# Main dataset file
DATASET_PATH = PROCESSED_DATA_DIR / "news_clean.csv"

# Dataset columns

TEXT_COLUMN = "content"

TITLE_COLUMN = "title"

SUMMARY_COLUMN = "summary"

CATEGORY_COLUMN = "category"

SUBCATEGORY_COLUMN = "sub_category"

DATE_COLUMN = "published date"



# =========================================================
# Experiment Configuration
# =========================================================

# Used for naming output files

EXPERIMENT_NAME = "baseline"

# Change to:
# "finetuned"
# when running the fine-tuned model.



# =========================================================
# Random Seed
# =========================================================

RANDOM_STATE = 42



# =========================================================
# Embedding Model Configuration
# =========================================================

EMBEDDING_MODEL_NAME = "HooshvareLab/bert-base-parsbert-uncased"

MAX_LENGTH = 512

BATCH_SIZE = 16 # maybe need to change



# =========================================================
# UMAP Configuration
# Used in BERTopic
# =========================================================

N_NEIGHBORS = 15

N_COMPONENTS = 5

MIN_DIST = 0.0

UMAP_METRIC = "cosine"



# =========================================================
# HDBSCAN Configuration
# =========================================================

MIN_CLUSTER_SIZE = 100

MIN_SAMPLES = 10

CLUSTER_SELECTION_METHOD = "eom"

PREDICTION_DATA = True



# =========================================================
# CountVectorizer Configuration
# =========================================================

TOP_N_WORDS = 10

NGRAM_RANGE = (1, 2)



# =========================================================
# BERTopic Configuration
# =========================================================

CALCULATE_PROBABILITIES = True

VERBOSE = True

LANGUAGE = None



# =========================================================
# Evaluation Configuration
# Used in evaluation.py
# =========================================================

COHERENCE_METHOD = "c_v"
COHERENCE_SIZE_SAMPLE = 1194

TOP_N_DIVERSITY = 10



# =========================================================
# Visualization Configuration
# Used in visualization.py
# =========================================================

FIGURE_WIDTH = 1200

FIGURE_HEIGHT = 800

FIGURE_SIZE = (12, 6)

FIGURE_DPI = 300

FIGURE_FORMAT = "png"



# =========================================================
# Saving Configuration
# =========================================================

CSV_ENCODING = "utf-8-sig"



# =========================================================
# Fine-Tuning Configuration
# =========================================================

NEGATIVE_STRATEGY = "easy"

FINETUNED_MODEL_DIR = (
    MODELS_DIR / 
    "finetuned" /
    "batch_hard_triplet"
)


FINETUNE_MODEL_NAME = EMBEDDING_MODEL_NAME

LEARNING_RATE = 2e-5

EPOCHS = 8

TRAIN_BATCH_SIZE = 16

WEIGHT_DECAY = 0.01

MAX_SEQ_LENGTH = 256


# --------------------------------------------------
# Topic Labeling
# --------------------------------------------------

OLLAMA_MODEL = "llama3.1:8b"

TOPIC_LABEL_MAX_WORDS = 4

N_TOPIC_WORDS = 12