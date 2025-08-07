"""
BBC News Dataset Processor for Real Kaggle Data

This script processes the actual BBC News dataset from Kaggle and prepares it
for topic modeling analysis.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def extract_category_from_url(url):
    """
    Extract category from BBC URL.
    
    Args:
        url: BBC news URL
        
    Returns:
        str: Extracted category or 'general'
    """
    if pd.isna(url):
        return 'general'
    
    # Common BBC news categories
    categories = {
        'business': 'business',
        'sport': 'sport', 
        'technology': 'technology',
        'entertainment': 'entertainment',
        'politics': 'politics',
        'health': 'health',
        'science': 'science',
        'education': 'education',
        'world': 'world',
        'uk': 'uk_news'
    }
    
    url_lower = url.lower()
    for category_key, category_name in categories.items():
        if f'/{category_key}/' in url_lower or f'/{category_key}-' in url_lower:
            return category_name
    
    return 'general'


def load_and_process_bbc_data(csv_path, sample_size=None):
    """
    Load and process the BBC News dataset.
    
    Args:
        csv_path: Path to the CSV file
        sample_size: Optional sample size for testing (None for full dataset)
        
    Returns:
        pd.DataFrame: Processed dataset
    """
    print(f"Loading BBC News dataset from {csv_path}...")
    
    # Load the dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} articles")
    
    # Check columns
    print(f"Columns: {list(df.columns)}")
    
    # Extract categories from URLs
    if 'link' in df.columns:
        print("Extracting categories from URLs...")
        df['category'] = df['link'].apply(extract_category_from_url)
    else:
        print("No 'link' column found, setting category as 'general'")
        df['category'] = 'general'
    
    # Combine title and description for text analysis
    if 'title' in df.columns and 'description' in df.columns:
        print("Combining title and description...")
        df['text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
    elif 'description' in df.columns:
        df['text'] = df['description'].fillna('')
    elif 'title' in df.columns:
        df['text'] = df['title'].fillna('')
    else:
        raise ValueError("No text content found in dataset")
    
    # Remove empty texts
    df = df[df['text'].str.len() > 10].copy()
    print(f"After removing short texts: {len(df)} articles")
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} articles for testing...")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    # Display category distribution
    print(f"\nCategory distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    return df


def analyze_dataset(df):
    """
    Analyze the BBC News dataset.
    
    Args:
        df: Processed DataFrame
    """
    print(f"\n📊 Dataset Analysis:")
    print(f"=" * 50)
    print(f"Total articles: {len(df)}")
    print(f"Date range: {df['pubDate'].min()} to {df['pubDate'].max()}")
    print(f"Average text length: {df['text'].str.len().mean():.0f} characters")
    print(f"Text length range: {df['text'].str.len().min()} - {df['text'].str.len().max()}")
    
    # Text length statistics by category
    print(f"\n📈 Text Length by Category:")
    text_stats = df.groupby('category')['text'].agg(['count', 'mean', 'std']).round(0)
    text_stats.columns = ['Count', 'Avg_Length', 'Std_Length']
    print(text_stats)
    
    # Sample articles from each category
    print(f"\n📄 Sample Articles by Category:")
    for category in df['category'].unique()[:5]:  # Show first 5 categories
        sample = df[df['category'] == category].iloc[0]
        print(f"\n🏷️ {category.upper()}:")
        print(f"Title: {sample.get('title', 'N/A')}")
        print(f"Text preview: {sample['text'][:150]}...")


def visualize_dataset(df):
    """
    Create visualizations for the dataset.
    
    Args:
        df: Processed DataFrame
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Category distribution
    category_counts = df['category'].value_counts()
    axes[0, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Distribution of News Categories')
    
    # Text length distribution
    axes[0, 1].hist(df['text'].str.len(), bins=50, alpha=0.7, color='skyblue')
    axes[0, 1].set_xlabel('Text Length (characters)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Article Lengths')
    
    # Articles over time
    df['date'] = pd.to_datetime(df['pubDate'])
    df['date_only'] = df['date'].dt.date
    daily_counts = df['date_only'].value_counts().sort_index()
    axes[1, 0].plot(daily_counts.index, daily_counts.values)
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Number of Articles')
    axes[1, 0].set_title('Articles Published Over Time')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Text length by category
    categories = df['category'].value_counts().head(8).index
    category_lengths = [df[df['category'] == cat]['text'].str.len() for cat in categories]
    axes[1, 1].boxplot(category_lengths, labels=categories)
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Text Length')
    axes[1, 1].set_title('Text Length Distribution by Category')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()


def save_processed_data(df, output_path):
    """
    Save the processed dataset.
    
    Args:
        df: Processed DataFrame
        output_path: Output file path
    """
    # Select relevant columns
    columns_to_save = ['text', 'category', 'title', 'pubDate']
    df_save = df[columns_to_save].copy()
    
    df_save.to_csv(output_path, index=False)
    print(f"✅ Processed dataset saved to {output_path}")


if __name__ == "__main__":
    # Process the BBC News dataset
    data_path = "../data/bbc_news.csv"
    
    try:
        # Load and process (use sample for initial testing)
        df = load_and_process_bbc_data(data_path, sample_size=1000)
        
        # Analyze the dataset
        analyze_dataset(df)
        
        # Create visualizations
        visualize_dataset(df)
        
        # Save processed data
        save_processed_data(df, "../data/processed_bbc_news.csv")
        
        print(f"\n🎉 Dataset processing completed!")
        print(f"Ready for topic modeling with {len(df)} articles across {df['category'].nunique()} categories")
        
    except Exception as e:
        print(f"❌ Error processing dataset: {e}")
        print("Please check the dataset path and format")
