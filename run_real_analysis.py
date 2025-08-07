"""
Real BBC News Dataset Analysis for Topic Modeling

This script demonstrates topic modeling on the actual BBC News dataset
with comprehensive analysis and visualization.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Import our custom modules
try:
    from data_loader import BBCNewsLoader, get_dataset_info
    from text_preprocessor import TextPreprocessor
    from topic_modeling import LDATopicModeler, NMFTopicModeler, compare_models
    from visualizations import create_wordcloud, plot_topic_words
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)

def extract_category_from_url(url):
    """Extract category from BBC URL."""
    if pd.isna(url):
        return 'general'
    
    categories = ['business', 'sport', 'technology', 'entertainment', 'politics', 'health', 'science', 'world', 'uk']
    url_lower = str(url).lower()
    
    for cat in categories:
        if f'/{cat}/' in url_lower or f'/{cat}-' in url_lower:
            return cat
    return 'general'

def main():
    print("🔄 BBC News Topic Modeling Analysis")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("\n📥 Step 1: Loading BBC News Dataset")
    df = pd.read_csv('data/bbc_news.csv')
    print(f"✅ Loaded {len(df)} articles")
    
    # Extract categories and combine text
    df['category'] = df['link'].apply(extract_category_from_url)
    df['text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
    df = df[df['text'].str.len() > 50].copy()  # Filter short texts
    
    # Sample for demonstration (remove this line to use full dataset)
    df = df.sample(n=2000, random_state=42).reset_index(drop=True)
    
    print(f"📊 Dataset Overview:")
    print(f"Articles after filtering: {len(df)}")
    print(f"Categories: {df['category'].value_counts()}")
    
    # Step 2: Text Preprocessing
    print("\n🔧 Step 2: Text Preprocessing")
    preprocessor = TextPreprocessor(use_spacy=True)
    
    # Preprocess texts
    texts = df['text'].tolist()
    processed_docs = preprocessor.preprocess_corpus(texts, min_doc_freq=3, max_doc_freq=0.8)
    processed_texts_nmf = [' '.join(doc) for doc in processed_docs]
    
    print(f"✅ Preprocessing completed")
    print(f"Total tokens: {sum(len(doc) for doc in processed_docs)}")
    print(f"Average tokens per document: {np.mean([len(doc) for doc in processed_docs]):.1f}")
    
    # Step 3: LDA Topic Modeling
    print("\n🤖 Step 3: LDA Topic Modeling")
    num_topics = min(8, df['category'].nunique())  # Adjust based on categories
    
    lda_model = LDATopicModeler(num_topics=num_topics, random_state=42)
    lda_model.prepare_corpus(processed_docs)
    lda_model.train_model(iterations=50, passes=5)  # Reduced for faster demo
    
    lda_topics = lda_model.get_topics(num_words=10)
    lda_coherence = lda_model.calculate_coherence(processed_docs)
    
    print(f"✅ LDA training completed")
    print(f"Coherence score: {lda_coherence:.4f}")
    
    # Step 4: NMF Topic Modeling
    print("\n🤖 Step 4: NMF Topic Modeling")
    nmf_model = NMFTopicModeler(num_topics=num_topics, random_state=42, use_tfidf=True)
    nmf_model.prepare_corpus(processed_texts_nmf)
    nmf_model.train_model(max_iter=100)
    
    nmf_topics = nmf_model.get_topics(num_words=10)
    nmf_coherence = nmf_model.calculate_coherence()
    
    print(f"✅ NMF training completed")
    print(f"Coherence score: {nmf_coherence:.4f}")
    
    # Step 5: Display Results
    print("\n🎯 Step 5: Topic Analysis Results")
    print("\n" + "="*60)
    print("LDA TOPICS:")
    print("="*60)
    for topic in lda_topics:
        print(f"\nTopic {topic['topic_id']}: {', '.join(topic['words'][:8])}")
    
    print("\n" + "="*60)
    print("NMF TOPICS:")
    print("="*60)
    for topic in nmf_topics:
        print(f"\nTopic {topic['topic_id']}: {', '.join(topic['words'][:8])}")
    
    # Step 6: Model Comparison
    print(f"\n📊 Step 6: Model Comparison")
    comparison = compare_models(lda_topics, nmf_topics)
    
    print(f"\nPerformance Comparison:")
    print(f"LDA Coherence: {lda_coherence:.4f}")
    print(f"NMF Coherence: {nmf_coherence:.4f}")
    
    print(f"\nTopic Overlap Analysis:")
    for overlap in comparison['topic_overlap']:
        print(f"LDA Topic {overlap['lda_topic']} ↔ NMF Topic {overlap['nmf_topic']}: {overlap['overlap_count']}/5 words overlap")
    
    # Step 7: Visualizations
    print(f"\n🎨 Step 7: Creating Visualizations")
    
    # Create topic words visualization
    try:
        fig = plot_topic_words(lda_topics[:4], num_words=8)  # Show first 4 topics
        plt.savefig('visualizations/lda_topics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Topic words plot created")
    except Exception as e:
        print(f"❌ Error creating topic plot: {e}")
    
    # Create word clouds
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, topic in enumerate(lda_topics[:4]):
            word_freq = dict(zip(topic['words'][:15], topic['weights'][:15]))
            
            from wordcloud import WordCloud
            wordcloud = WordCloud(width=300, height=200, background_color='white',
                                colormap='viridis', relative_scaling=0.5).generate_from_frequencies(word_freq)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].axis('off')
            axes[i].set_title(f'LDA Topic {i}', fontweight='bold')
        
        plt.suptitle('LDA Topics - Word Clouds', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/lda_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Word clouds created")
    except Exception as e:
        print(f"❌ Error creating word clouds: {e}")
    
    # Step 8: Document Analysis
    print(f"\n📄 Step 8: Document-Topic Analysis")
    
    # Analyze sample documents
    for i in range(min(3, len(df))):
        print(f"\n📋 Document {i+1}:")
        print(f"Category: {df.iloc[i]['category']}")
        print(f"Title: {df.iloc[i]['title']}")
        
        # Get topic distribution
        doc_topics = lda_model.get_document_topics(processed_docs[i])
        print(f"LDA Topics: ", end="")
        for topic_id, prob in sorted(doc_topics, key=lambda x: x[1], reverse=True)[:3]:
            print(f"Topic {topic_id}({prob:.2f}) ", end="")
        print()
    
    # Step 9: Category-Topic Analysis
    print(f"\n🏷️ Step 9: Category-Topic Relationship")
    
    # Create document-topic matrix
    doc_topic_matrix = np.zeros((len(df), num_topics))
    for i, doc in enumerate(processed_docs):
        doc_topics = lda_model.get_document_topics(doc)
        for topic_id, prob in doc_topics:
            doc_topic_matrix[i, topic_id] = prob
    
    # Average by category
    category_topics = pd.DataFrame(doc_topic_matrix, columns=[f'Topic_{i}' for i in range(num_topics)])
    category_topics['category'] = df['category'].values
    category_topic_avg = category_topics.groupby('category').mean()
    
    print("Average topic distribution by category:")
    print(category_topic_avg.round(3))
    
    # Save results
    print(f"\n💾 Step 10: Saving Results")
    
    # Save models
    lda_model.save_model('results/lda_model.pkl')
    nmf_model.save_model('results/nmf_model.pkl')
    
    # Save processed data
    df_results = df.copy()
    df_results['processed_text'] = processed_texts_nmf
    df_results.to_csv('results/processed_bbc_news.csv', index=False)
    
    print(f"✅ Results saved to results/ directory")
    
    print(f"\n🎉 BBC News Topic Modeling Analysis Completed!")
    print(f"📊 Summary:")
    print(f"  - Analyzed {len(df)} articles across {df['category'].nunique()} categories")
    print(f"  - Extracted {num_topics} topics using LDA and NMF")
    print(f"  - LDA Coherence: {lda_coherence:.4f}")
    print(f"  - NMF Coherence: {nmf_coherence:.4f}")
    print(f"  - Visualizations saved to visualizations/ directory")
    print(f"  - Models and data saved to results/ directory")

if __name__ == "__main__":
    main()
