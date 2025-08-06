"""
Data Loading and Preprocessing Utilities for Topic Modeling

This module provides functions to load and preprocess the BBC News dataset
for topic modeling analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import List, Tuple, Dict


class BBCNewsLoader:
    """Utility class for loading BBC News dataset from various formats."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    
    def load_from_folders(self) -> pd.DataFrame:
        """
        Load BBC News data from category folders structure.
        
        Expected structure:
        data/
        ├── business/
        │   ├── article1.txt
        │   └── article2.txt
        ├── entertainment/
        └── ...
        
        Returns:
            pd.DataFrame: DataFrame with columns ['text', 'category', 'filename']
        """
        articles = []
        
        for category in self.categories:
            category_path = self.data_dir / category
            if category_path.exists():
                for file_path in category_path.glob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        
                        articles.append({
                            'text': text,
                            'category': category,
                            'filename': file_path.name
                        })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        return pd.DataFrame(articles)
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load BBC News data from CSV file.
        
        Expected columns: text, category (and optionally title, filename)
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist
            required_cols = ['text', 'category']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            return df
        except Exception as e:
            raise Exception(f"Error loading CSV: {e}")
    
    def auto_load(self) -> pd.DataFrame:
        """
        Automatically detect and load BBC News dataset from available format.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        # First try to find CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if csv_files:
            print(f"Found CSV file: {csv_files[0]}")
            return self.load_from_csv(csv_files[0])
        
        # Then try folder structure
        category_folders = [self.data_dir / cat for cat in self.categories]
        if any(folder.exists() for folder in category_folders):
            print("Found category folders structure")
            return self.load_from_folders()
        
        raise FileNotFoundError(
            "No BBC News dataset found. Please ensure data is in one of these formats:\n"
            "1. CSV file in data/ directory with 'text' and 'category' columns\n"
            "2. Folder structure: data/category_name/*.txt files"
        )


def basic_text_cleaning(text: str) -> str:
    """
    Perform basic text cleaning operations.
    
    Args:
        text: Raw text string
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?]', '', text)
    
    # Remove very short words (likely not meaningful)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    
    return ' '.join(words).strip()


def get_dataset_info(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive information about the loaded dataset.
    
    Args:
        df: Dataset DataFrame
        
    Returns:
        dict: Dataset statistics and information
    """
    info = {
        'total_articles': len(df),
        'categories': df['category'].value_counts().to_dict(),
        'avg_text_length': df['text'].str.len().mean(),
        'min_text_length': df['text'].str.len().min(),
        'max_text_length': df['text'].str.len().max(),
        'columns': list(df.columns)
    }
    
    return info


if __name__ == "__main__":
    # Example usage
    loader = BBCNewsLoader()
    try:
        df = loader.auto_load()
        print("Dataset loaded successfully!")
        
        info = get_dataset_info(df)
        print(f"Total articles: {info['total_articles']}")
        print(f"Categories: {info['categories']}")
        print(f"Average text length: {info['avg_text_length']:.1f} characters")
        
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
