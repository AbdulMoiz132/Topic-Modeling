"""
Visualization utilities for topic modeling results.

This module provides functions to create various visualizations for topic modeling,
including word clouds, topic distributions, and interactive pyLDAvis plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from .config import FIGSIZE, WORDS_PER_TOPIC, VISUALIZATIONS_DIR


def create_wordcloud(topic_words: List[str], topic_weights: List[float],
                    title: str = "Topic Word Cloud", figsize: tuple = FIGSIZE) -> plt.Figure:
    """
    Create a word cloud for a topic.
    
    Args:
        topic_words: List of words in the topic
        topic_weights: List of word weights/probabilities
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        plt.Figure: Word cloud figure
    """
    # Create word frequency dictionary
    word_freq = dict(zip(topic_words, topic_weights))
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, height=400,
        background_color='white',
        max_words=50,
        colormap='viridis',
        relative_scaling=0.5
    ).generate_from_frequencies(word_freq)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_topic_words(topics: List[Dict], num_words: int = WORDS_PER_TOPIC,
                    figsize: tuple = (15, 10)) -> plt.Figure:
    """
    Create a horizontal bar plot showing top words for each topic.
    
    Args:
        topics: List of topic dictionaries
        num_words: Number of top words to show per topic
        figsize: Figure size
        
    Returns:
        plt.Figure: Bar plot figure
    """
    num_topics = len(topics)
    cols = 2
    rows = (num_topics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, topic in enumerate(topics):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        words = topic['words'][:num_words]
        weights = topic['weights'][:num_words]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(words))
        bars = ax.barh(y_pos, weights, color=plt.cm.viridis(i / num_topics))
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(f'Topic {topic["topic_id"]}', fontweight='bold')
        
        # Add value labels on bars
        for j, (bar, weight) in enumerate(zip(bars, weights)):
            ax.text(weight + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{weight:.3f}', va='center', fontsize=8)
    
    # Hide empty subplots
    for i in range(num_topics, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_topic_distribution(doc_topics: List[List[tuple]], 
                          category_labels: Optional[List[str]] = None,
                          figsize: tuple = FIGSIZE) -> plt.Figure:
    """
    Plot topic distribution across documents or categories.
    
    Args:
        doc_topics: List of topic distributions for each document
        category_labels: Optional category labels for documents
        figsize: Figure size
        
    Returns:
        plt.Figure: Distribution plot figure
    """
    # Convert to matrix format
    num_docs = len(doc_topics)
    num_topics = max(max(topic_id for topic_id, _ in doc) for doc in doc_topics) + 1
    
    topic_matrix = np.zeros((num_docs, num_topics))
    for doc_idx, doc_topic_dist in enumerate(doc_topics):
        for topic_id, prob in doc_topic_dist:
            topic_matrix[doc_idx, topic_id] = prob
    
    if category_labels:
        # Plot by category
        df = pd.DataFrame(topic_matrix, columns=[f'Topic {i}' for i in range(num_topics)])
        df['Category'] = category_labels
        
        # Calculate average topic distribution per category
        category_topics = df.groupby('Category').mean()
        
        fig, ax = plt.subplots(figsize=figsize)
        category_topics.plot(kind='bar', ax=ax, colormap='viridis')
        ax.set_title('Average Topic Distribution by Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Average Topic Probability')
        ax.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
    else:
        # Plot overall distribution
        fig, ax = plt.subplots(figsize=figsize)
        topic_means = topic_matrix.mean(axis=0)
        topic_stds = topic_matrix.std(axis=0)
        
        x = np.arange(num_topics)
        bars = ax.bar(x, topic_means, yerr=topic_stds, capsize=5, 
                     color=plt.cm.viridis(x / num_topics))
        
        ax.set_xlabel('Topic ID')
        ax.set_ylabel('Average Probability')
        ax.set_title('Average Topic Distribution Across Documents', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Topic {i}' for i in range(num_topics)])
    
    plt.tight_layout()
    return fig


def create_interactive_topic_plot(topics: List[Dict]) -> go.Figure:
    """
    Create an interactive plotly visualization of topics.
    
    Args:
        topics: List of topic dictionaries
        
    Returns:
        go.Figure: Interactive plotly figure
    """
    # Prepare data for heatmap
    all_words = set()
    for topic in topics:
        all_words.update(topic['words'][:WORDS_PER_TOPIC])
    
    all_words = sorted(list(all_words))
    
    # Create weight matrix
    weight_matrix = []
    topic_labels = []
    
    for topic in topics:
        topic_weights = []
        word_weight_dict = dict(topic['word_weight_pairs'])
        
        for word in all_words:
            topic_weights.append(word_weight_dict.get(word, 0))
        
        weight_matrix.append(topic_weights)
        topic_labels.append(f"Topic {topic['topic_id']}")
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=weight_matrix,
        x=all_words,
        y=topic_labels,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Topic: %{y}<br>Word: %{x}<br>Weight: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Topic-Word Weight Heatmap',
        xaxis_title='Words',
        yaxis_title='Topics',
        width=1200,
        height=600
    )
    
    return fig


def create_pyldavis_visualization(lda_model, corpus, dictionary, 
                                save_path: Optional[str] = None) -> str:
    """
    Create pyLDAvis interactive visualization for LDA model.
    
    Args:
        lda_model: Trained Gensim LDA model
        corpus: Gensim corpus
        dictionary: Gensim dictionary
        save_path: Optional path to save HTML file
        
    Returns:
        str: HTML content or file path
    """
    # Prepare pyLDAvis data
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)
    
    if save_path:
        pyLDAvis.save_html(vis_data, save_path)
        return save_path
    else:
        return pyLDAvis.prepared_data_to_html(vis_data)


def plot_model_comparison(lda_topics: List[Dict], nmf_topics: List[Dict],
                         figsize: tuple = (15, 8)) -> plt.Figure:
    """
    Create a comparison plot between LDA and NMF topics.
    
    Args:
        lda_topics: Topics from LDA model
        nmf_topics: Topics from NMF model
        figsize: Figure size
        
    Returns:
        plt.Figure: Comparison plot figure
    """
    num_topics = len(lda_topics)
    
    fig, axes = plt.subplots(2, num_topics, figsize=figsize)
    
    for i in range(num_topics):
        # LDA topics (top row)
        lda_words = lda_topics[i]['words'][:5]
        lda_weights = lda_topics[i]['weights'][:5]
        
        axes[0, i].barh(range(len(lda_words)), lda_weights, color='skyblue')
        axes[0, i].set_yticks(range(len(lda_words)))
        axes[0, i].set_yticklabels(lda_words)
        axes[0, i].invert_yaxis()
        axes[0, i].set_title(f'LDA Topic {i}', fontweight='bold')
        if i == 0:
            axes[0, i].set_ylabel('LDA', fontweight='bold')
        
        # NMF topics (bottom row)
        nmf_words = nmf_topics[i]['words'][:5]
        nmf_weights = nmf_topics[i]['weights'][:5]
        
        axes[1, i].barh(range(len(nmf_words)), nmf_weights, color='lightcoral')
        axes[1, i].set_yticks(range(len(nmf_words)))
        axes[1, i].set_yticklabels(nmf_words)
        axes[1, i].invert_yaxis()
        axes[1, i].set_title(f'NMF Topic {i}', fontweight='bold')
        if i == 0:
            axes[1, i].set_ylabel('NMF', fontweight='bold')
    
    plt.suptitle('LDA vs NMF Topic Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def save_all_visualizations(topics: List[Dict], output_dir: str = VISUALIZATIONS_DIR) -> None:
    """
    Save all topic visualizations to files.
    
    Args:
        topics: List of topic dictionaries
        output_dir: Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save word clouds for each topic
    for topic in topics:
        fig = create_wordcloud(
            topic['words'], 
            topic['weights'],
            title=f"Topic {topic['topic_id']} Word Cloud"
        )
        fig.savefig(f"{output_dir}/topic_{topic['topic_id']}_wordcloud.png", 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # Save topic words bar plot
    fig = plot_topic_words(topics)
    fig.savefig(f"{output_dir}/topic_words_barplot.png", 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save interactive plot as HTML
    interactive_fig = create_interactive_topic_plot(topics)
    interactive_fig.write_html(f"{output_dir}/interactive_topics.html")
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage with dummy data
    dummy_topics = [
        {
            'topic_id': 0,
            'words': ['business', 'company', 'market', 'economy', 'financial'],
            'weights': [0.15, 0.12, 0.10, 0.08, 0.07],
            'word_weight_pairs': [('business', 0.15), ('company', 0.12), ('market', 0.10)]
        },
        {
            'topic_id': 1,
            'words': ['sports', 'game', 'team', 'player', 'match'],
            'weights': [0.18, 0.14, 0.11, 0.09, 0.08],
            'word_weight_pairs': [('sports', 0.18), ('game', 0.14), ('team', 0.11)]
        }
    ]
    
    # Create sample visualizations
    fig = create_wordcloud(dummy_topics[0]['words'], dummy_topics[0]['weights'])
    plt.show()
    
    fig = plot_topic_words(dummy_topics)
    plt.show()
