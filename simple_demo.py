"""
BBC News Topic Modeling Demo

Quick demonstration and validation of the topic modeling pipeline
with the real BBC News dataset.
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('src')

def validate_dataset():
    """Quick validation of the BBC News dataset."""
    print("🔄 BBC News Dataset Validation")
    print("=" * 40)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('data/bbc_news.csv')
    print(f"✅ Loaded {len(df)} articles")
    
    # Extract categories from URLs
    def get_category(url):
        if pd.isna(url): return 'general'
        url = str(url).lower()
        categories = {
            'business': 'business',
            'sport': 'sport', 
            'technology': 'technology',
            'entertainment': 'entertainment',
            'politics': 'politics',
            'health': 'health',
            'science': 'science',
            'world': 'world',
            'uk': 'uk_news'
        }
        for key, value in categories.items():
            if f'/{key}/' in url or f'/{key}-' in url:
                return value
        return 'general'
    
    df['category'] = df['link'].apply(get_category)
    df['text'] = df['title'].fillna('') + '. ' + df['description'].fillna('')
    
    print(f"📊 Dataset Statistics:")
    print(f"  - Total articles: {len(df)}")
    print(f"  - Average text length: {df['text'].str.len().mean():.0f} characters")
    print(f"  - Categories found: {df['category'].value_counts().head().to_dict()}")
    
    # Show sample articles
    print(f"\n📄 Sample Articles:")
    for i, category in enumerate(df['category'].value_counts().head(3).index):
        sample = df[df['category'] == category].iloc[0]
        print(f"\n{i+1}. {category.upper()}:")
        print(f"   Title: {sample['title']}")
        print(f"   Text: {sample['text'][:120]}...")
    
    print(f"\n✅ Dataset validation completed!")
    print(f"🚀 Ready for topic modeling analysis!")
    
    return df

if __name__ == "__main__":
    validate_dataset()
