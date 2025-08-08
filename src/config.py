"""
Configuration settings for the Topic Modeling project.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "visualizations"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
VISUALIZATIONS_DIR.mkdir(exist_ok=True)

# Visualization settings
FIGSIZE = (12, 8)
WORDS_PER_TOPIC = 10

# Dataset settings
DATASET_FILE = DATA_DIR / "bbc_news.csv"

# Model settings
DEFAULT_NUM_TOPICS = 5
MIN_DOCUMENT_FREQUENCY = 2
MAX_DOCUMENT_FREQUENCY = 0.95
MAX_FEATURES = 5000

# Text preprocessing settings
CUSTOM_STOPWORDS = [
    'bbc', 'news', 'said', 'say', 'says', 'told', 'tells', 'new', 'uk',
    'britain', 'british', 'london', 'england', 'scotland', 'wales',
    'government', 'minister', 'ministers', 'prime', 'mr', 'ms', 'mrs',
    'one', 'two', 'three', 'first', 'second', 'third', 'last', 'also',
    'would', 'could', 'should', 'may', 'might', 'must', 'can', 'will',
    'time', 'year', 'years', 'day', 'days', 'week', 'weeks', 'month', 'months',
    'people', 'person', 'man', 'woman', 'men', 'women',
    'way', 'ways', 'part', 'parts', 'number', 'numbers'
]

# File paths for saved models
LDA_MODEL_PATH = RESULTS_DIR / "lda_model.pkl"
NMF_MODEL_PATH = RESULTS_DIR / "nmf_model.pkl"
PROCESSED_DATA_PATH = RESULTS_DIR / "processed_bbc_news.csv"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
