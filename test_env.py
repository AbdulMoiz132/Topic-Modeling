"""
Test script to verify the environment setup.
"""

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✓ pandas")
    except ImportError as e:
        print(f"✗ pandas: {e}")
    
    try:
        import numpy as np
        print("✓ numpy")
    except ImportError as e:
        print(f"✗ numpy: {e}")
    
    try:
        import sklearn
        print("✓ scikit-learn")
    except ImportError as e:
        print(f"✗ scikit-learn: {e}")
    
    try:
        import gensim
        print("✓ gensim")
    except ImportError as e:
        print(f"✗ gensim: {e}")
    
    try:
        import nltk
        print("✓ nltk")
    except ImportError as e:
        print(f"✗ nltk: {e}")
    
    try:
        import spacy
        print("✓ spacy")
    except ImportError as e:
        print(f"✗ spacy: {e}")
    
    print("\nEnvironment test completed!")

if __name__ == "__main__":
    test_imports()
