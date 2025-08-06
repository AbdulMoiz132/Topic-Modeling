"""
Setup script to download required NLTK data and spaCy models.
Run this script after installing the required packages.
"""

import nltk
import subprocess
import sys


def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    
    nltk_downloads = [
        'punkt',
        'stopwords', 
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]
    
    for item in nltk_downloads:
        try:
            print(f"Downloading {item}...")
            nltk.download(item, quiet=True)
            print(f"✓ {item} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {item}: {e}")


def download_spacy_model():
    """Download spaCy English model."""
    print("\nDownloading spaCy English model...")
    
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✓ spaCy English model already available")
        except OSError:
            print("Installing spaCy English model...")
            subprocess.run([
                sys.executable, "-m", "spacy", "download", "en_core_web_sm"
            ], check=True)
            print("✓ spaCy English model downloaded successfully")
    except Exception as e:
        print(f"✗ Error with spaCy model: {e}")


def verify_setup():
    """Verify that all components are working."""
    print("\nVerifying setup...")
    
    try:
        # Test NLTK
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        stops = stopwords.words('english')
        tokens = word_tokenize("This is a test.")
        print("✓ NLTK working correctly")
    except Exception as e:
        print(f"✗ NLTK error: {e}")
    
    try:
        # Test spaCy
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test.")
        print("✓ spaCy working correctly")
    except Exception as e:
        print(f"✗ spaCy error: {e}")
    
    try:
        # Test other imports
        import gensim
        import sklearn
        import pandas as pd
        import numpy as np
        print("✓ All main packages imported successfully")
    except Exception as e:
        print(f"✗ Package import error: {e}")


if __name__ == "__main__":
    print("Setting up Topic Modeling environment...\n")
    
    download_nltk_data()
    download_spacy_model()
    verify_setup()
    
    print("\n🎉 Setup completed! You're ready to start topic modeling.")
