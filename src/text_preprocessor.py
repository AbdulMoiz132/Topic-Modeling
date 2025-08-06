"""
Text Preprocessing Pipeline for Topic Modeling

This module provides comprehensive text preprocessing functionality
specifically designed for topic modeling tasks.
"""

import nltk
import spacy
import re
import string
from typing import List, Set, Optional
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline for topic modeling."""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the text preprocessor.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP (recommended)
        """
        self.use_spacy = use_spacy
        self._setup_nltk()
        
        if use_spacy:
            self._setup_spacy()
        else:
            self._setup_nltk_only()
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Combine NLTK and sklearn stopwords
        nltk_stops = set(stopwords.words('english'))
        sklearn_stops = set(ENGLISH_STOP_WORDS)
        self.stop_words = nltk_stops.union(sklearn_stops)
        
        # Add common news-specific stopwords
        news_stops = {
            'said', 'say', 'says', 'would', 'could', 'also', 'one', 'two', 
            'first', 'last', 'new', 'year', 'time', 'people', 'man', 'woman',
            'way', 'use', 'make', 'get', 'go', 'come', 'know', 'take', 'see',
            'think', 'back', 'good', 'want', 'give', 'well', 'work', 'part',
            'find', 'right', 'still', 'even', 'much', 'long', 'may', 'might'
        }
        self.stop_words.update(news_stops)
    
    def _setup_spacy(self):
        """Setup spaCy for advanced NLP processing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy English model not found. Installing...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def _setup_nltk_only(self):
        """Setup NLTK-only processing."""
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        Perform basic text cleaning.
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (but keep words with numbers)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except sentence endings
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize text and perform lemmatization.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        if self.use_spacy:
            return self._spacy_process(text)
        else:
            return self._nltk_process(text)
    
    def _spacy_process(self, text: str) -> List[str]:
        """Process text using spaCy."""
        doc = self.nlp(text)
        
        tokens = []
        for token in doc:
            # Skip stopwords, punctuation, spaces, and short tokens
            if (not token.is_stop and 
                not token.is_punct and 
                not token.is_space and 
                len(token.text) > 2 and
                token.text.lower() not in self.stop_words and
                token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']):  # Keep meaningful POS tags
                
                # Use lemma and convert to lowercase
                lemma = token.lemma_.lower().strip()
                if len(lemma) > 2 and lemma.isalpha():
                    tokens.append(lemma)
        
        return tokens
    
    def _nltk_process(self, text: str) -> List[str]:
        """Process text using NLTK."""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter and lemmatize
        processed_tokens = []
        for token in tokens:
            token = token.lower().strip()
            
            if (len(token) > 2 and 
                token.isalpha() and 
                token not in self.stop_words):
                
                # Lemmatize
                lemma = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemma)
        
        return processed_tokens
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw text string
            
        Returns:
            List[str]: List of preprocessed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize and lemmatize
        tokens = self.tokenize_and_lemmatize(cleaned)
        
        return tokens
    
    def preprocess_corpus(self, texts: List[str], min_doc_freq: int = 2, 
                         max_doc_freq: float = 0.8) -> List[List[str]]:
        """
        Preprocess an entire corpus with frequency filtering.
        
        Args:
            texts: List of raw text documents
            min_doc_freq: Minimum document frequency for tokens
            max_doc_freq: Maximum document frequency ratio for tokens
            
        Returns:
            List[List[str]]: List of preprocessed document token lists
        """
        # Preprocess all documents
        processed_docs = []
        for text in texts:
            tokens = self.preprocess_text(text)
            processed_docs.append(tokens)
        
        # Calculate document frequencies
        vocab_doc_freq = {}
        total_docs = len(processed_docs)
        
        for doc_tokens in processed_docs:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                vocab_doc_freq[token] = vocab_doc_freq.get(token, 0) + 1
        
        # Filter tokens based on document frequency
        filtered_docs = []
        for doc_tokens in processed_docs:
            filtered_tokens = []
            for token in doc_tokens:
                doc_freq = vocab_doc_freq[token]
                doc_freq_ratio = doc_freq / total_docs
                
                if doc_freq >= min_doc_freq and doc_freq_ratio <= max_doc_freq:
                    filtered_tokens.append(token)
            
            filtered_docs.append(filtered_tokens)
        
        return filtered_docs


def setup_nltk_data():
    """Download necessary NLTK data."""
    nltk_downloads = [
        'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'
    ]
    
    for item in nltk_downloads:
        try:
            nltk.data.find(f'tokenizers/{item}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{item}')
            except LookupError:
                try:
                    nltk.data.find(f'taggers/{item}')
                except LookupError:
                    print(f"Downloading {item}...")
                    nltk.download(item)


if __name__ == "__main__":
    # Setup NLTK data
    setup_nltk_data()
    
    # Example usage
    preprocessor = TextPreprocessor(use_spacy=True)
    
    sample_text = """
    This is a sample news article about technology and business.
    It contains various types of information that needs to be processed
    for topic modeling analysis. The article discusses recent developments
    in artificial intelligence and machine learning.
    """
    
    tokens = preprocessor.preprocess_text(sample_text)
    print("Preprocessed tokens:", tokens)
