"""
Topic Modeling Implementations using LDA and NMF

This module provides implementations for Latent Dirichlet Allocation (LDA)
and Non-negative Matrix Factorization (NMF) topic modeling algorithms.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path

# Gensim imports for LDA
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel

# Scikit-learn imports for NMF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .config import *
except ImportError:
    from config import *


class LDATopicModeler:
    """
    Latent Dirichlet Allocation topic modeling using Gensim.
    """
    
    def __init__(self, num_topics: int = DEFAULT_NUM_TOPICS, 
                 random_state: int = RANDOM_STATE):
        """
        Initialize LDA topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            random_state: Random seed for reproducibility
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.coherence_score = None
    
    def prepare_corpus(self, tokenized_docs: List[List[str]]) -> None:
        """
        Prepare corpus for LDA training.
        
        Args:
            tokenized_docs: List of tokenized documents
        """
        # Create dictionary
        self.dictionary = corpora.Dictionary(tokenized_docs)
        
        # Filter extremes to improve topic quality
        self.dictionary.filter_extremes(
            no_below=MIN_DOCUMENT_FREQUENCY,
            no_above=MAX_DOCUMENT_FREQUENCY,
            keep_n=10000
        )
        
        # Create corpus (bag of words representation)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]
        
        print(f"Dictionary size: {len(self.dictionary)}")
        print(f"Corpus size: {len(self.corpus)}")
    
    def train_model(self, iterations: int = LDA_ITERATIONS, 
                   passes: int = LDA_PASSES) -> None:
        """
        Train the LDA model.
        
        Args:
            iterations: Number of iterations for training
            passes: Number of passes through the corpus
        """
        if self.corpus is None:
            raise ValueError("Corpus not prepared. Call prepare_corpus() first.")
        
        print(f"Training LDA model with {self.num_topics} topics...")
        
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            iterations=iterations,
            passes=passes,
            alpha='auto',
            per_word_topics=True,
            eval_every=10
        )
        
        print("LDA training completed!")
    
    def calculate_coherence(self, tokenized_docs: List[List[str]]) -> float:
        """
        Calculate coherence score for the model.
        
        Args:
            tokenized_docs: Original tokenized documents
            
        Returns:
            float: Coherence score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        coherence_model = CoherenceModel(
            model=self.model,
            texts=tokenized_docs,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        
        self.coherence_score = coherence_model.get_coherence()
        return self.coherence_score
    
    def get_topics(self, num_words: int = WORDS_PER_TOPIC) -> List[Dict]:
        """
        Get topics with their top words and weights.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List[Dict]: List of topics with words and weights
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        topics = []
        for topic_id in range(self.num_topics):
            topic_words = self.model.show_topic(topic_id, topn=num_words)
            topics.append({
                'topic_id': topic_id,
                'words': [word for word, _ in topic_words],
                'weights': [weight for _, weight in topic_words],
                'word_weight_pairs': topic_words
            })
        
        return topics
    
    def get_document_topics(self, doc_tokens: List[str]) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document.
        
        Args:
            doc_tokens: Tokenized document
            
        Returns:
            List[Tuple[int, float]]: Topic probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        bow = self.dictionary.doc2bow(doc_tokens)
        return self.model.get_document_topics(bow)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'dictionary': self.dictionary,
            'num_topics': self.num_topics,
            'coherence_score': self.coherence_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.dictionary = model_data['dictionary']
        self.num_topics = model_data['num_topics']
        self.coherence_score = model_data.get('coherence_score')
        
        print(f"Model loaded from {filepath}")


class NMFTopicModeler:
    """
    Non-negative Matrix Factorization topic modeling using scikit-learn.
    """
    
    def __init__(self, num_topics: int = DEFAULT_NUM_TOPICS,
                 random_state: int = RANDOM_STATE,
                 use_tfidf: bool = True):
        """
        Initialize NMF topic modeler.
        
        Args:
            num_topics: Number of topics to extract
            random_state: Random seed for reproducibility
            use_tfidf: Whether to use TF-IDF or count vectorization
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.use_tfidf = use_tfidf
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.doc_topic_matrix = None
    
    def prepare_corpus(self, texts: List[str]) -> None:
        """
        Prepare corpus for NMF training.
        
        Args:
            texts: List of preprocessed text documents (space-separated tokens)
        """
        # Choose vectorizer
        if self.use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_df=MAX_DOCUMENT_FREQUENCY,
                min_df=MIN_DOCUMENT_FREQUENCY,
                stop_words='english',
                lowercase=True,
                max_features=10000
            )
        else:
            self.vectorizer = CountVectorizer(
                max_df=MAX_DOCUMENT_FREQUENCY,
                min_df=MIN_DOCUMENT_FREQUENCY,
                stop_words='english',
                lowercase=True,
                max_features=10000
            )
        
        # Fit and transform documents
        self.doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Document-term matrix shape: {self.doc_term_matrix.shape}")
        print(f"Vocabulary size: {len(self.feature_names)}")
    
    def train_model(self, max_iter: int = 200) -> None:
        """
        Train the NMF model.
        
        Args:
            max_iter: Maximum number of iterations
        """
        if self.doc_term_matrix is None:
            raise ValueError("Corpus not prepared. Call prepare_corpus() first.")
        
        print(f"Training NMF model with {self.num_topics} topics...")
        
        self.model = NMF(
            n_components=self.num_topics,
            random_state=self.random_state,
            max_iter=max_iter,
            alpha_W=0.1,
            alpha_H=0.1,
            l1_ratio=0.5
        )
        
        # Fit model and get document-topic matrix
        self.doc_topic_matrix = self.model.fit_transform(self.doc_term_matrix)
        
        print("NMF training completed!")
    
    def get_topics(self, num_words: int = WORDS_PER_TOPIC) -> List[Dict]:
        """
        Get topics with their top words and weights.
        
        Args:
            num_words: Number of top words per topic
            
        Returns:
            List[Dict]: List of topics with words and weights
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        topics = []
        
        for topic_id in range(self.num_topics):
            # Get topic-word weights
            topic_weights = self.model.components_[topic_id]
            
            # Get top word indices
            top_word_indices = topic_weights.argsort()[-num_words:][::-1]
            
            # Get words and their weights
            words = [self.feature_names[i] for i in top_word_indices]
            weights = [topic_weights[i] for i in top_word_indices]
            
            topics.append({
                'topic_id': topic_id,
                'words': words,
                'weights': weights,
                'word_weight_pairs': list(zip(words, weights))
            })
        
        return topics
    
    def get_document_topics(self, doc_index: int) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document.
        
        Args:
            doc_index: Index of the document
            
        Returns:
            List[Tuple[int, float]]: Topic probabilities
        """
        if self.doc_topic_matrix is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        doc_topics = self.doc_topic_matrix[doc_index]
        # Normalize to get probabilities
        doc_topics = doc_topics / doc_topics.sum()
        
        return [(i, prob) for i, prob in enumerate(doc_topics)]
    
    def calculate_coherence(self) -> float:
        """
        Calculate a simple coherence measure for NMF.
        
        Returns:
            float: Average cosine similarity between top words
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        coherence_scores = []
        
        for topic_id in range(self.num_topics):
            topic_weights = self.model.components_[topic_id]
            top_word_indices = topic_weights.argsort()[-10:][::-1]
            
            # Get word vectors for top words
            word_vectors = self.doc_term_matrix[:, top_word_indices].T.toarray()
            
            # Calculate pairwise cosine similarities
            similarities = cosine_similarity(word_vectors)
            
            # Average similarity (excluding diagonal)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            avg_similarity = similarities[mask].mean()
            coherence_scores.append(avg_similarity)
        
        return np.mean(coherence_scores)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'num_topics': self.num_topics,
            'use_tfidf': self.use_tfidf,
            'feature_names': self.feature_names,
            'doc_topic_matrix': self.doc_topic_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.num_topics = model_data['num_topics']
        self.use_tfidf = model_data['use_tfidf']
        self.feature_names = model_data['feature_names']
        self.doc_topic_matrix = model_data['doc_topic_matrix']
        
        print(f"Model loaded from {filepath}")


def compare_models(lda_topics: List[Dict], nmf_topics: List[Dict]) -> Dict:
    """
    Compare LDA and NMF topic models.
    
    Args:
        lda_topics: Topics from LDA model
        nmf_topics: Topics from NMF model
        
    Returns:
        Dict: Comparison results
    """
    comparison = {
        'num_topics': len(lda_topics),
        'lda_topics': lda_topics,
        'nmf_topics': nmf_topics,
        'topic_overlap': []
    }
    
    # Calculate topic overlap (simplified)
    for lda_topic in lda_topics:
        lda_words = set(lda_topic['words'][:5])  # Top 5 words
        
        best_overlap = 0
        best_nmf_topic = None
        
        for nmf_topic in nmf_topics:
            nmf_words = set(nmf_topic['words'][:5])
            overlap = len(lda_words.intersection(nmf_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_nmf_topic = nmf_topic['topic_id']
        
        comparison['topic_overlap'].append({
            'lda_topic': lda_topic['topic_id'],
            'nmf_topic': best_nmf_topic,
            'overlap_count': best_overlap,
            'overlap_ratio': best_overlap / 5.0
        })
    
    return comparison
