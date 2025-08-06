"""
Configuration settings for the Topic Modeling project.
"""

# Data settings
DATA_DIR = "data"
RESULTS_DIR = "results"
VISUALIZATIONS_DIR = "visualizations"

# Text preprocessing settings
MIN_DOCUMENT_FREQUENCY = 2  # Minimum times a word should appear across documents
MAX_DOCUMENT_FREQUENCY = 0.8  # Maximum proportion of documents a word can appear in
MIN_WORD_LENGTH = 3  # Minimum word length to consider
USE_SPACY = True  # Whether to use spaCy for advanced NLP

# Topic modeling settings
DEFAULT_NUM_TOPICS = 5  # Number of topics for LDA/NMF
LDA_ITERATIONS = 1000  # Number of iterations for LDA training
LDA_PASSES = 10  # Number of passes through the corpus
RANDOM_STATE = 42  # For reproducible results

# Visualization settings
WORDS_PER_TOPIC = 10  # Number of top words to show per topic
FIGSIZE = (12, 8)  # Default figure size for plots

# File paths
BBC_NEWS_CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']
