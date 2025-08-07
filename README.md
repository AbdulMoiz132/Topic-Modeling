# Topic Modeling on BBC News Articles

This project implements comprehensive topic modeling analysis using both **Latent Dirichlet Allocation (LDA)** and **Non-negative Matrix Factorization (NMF)** algorithms on the BBC News dataset from Kaggle.

## 🎯 Project Overview

- **Dataset**: BBC News Dataset (42,000+ articles) with automatic category extraction
- **Algorithms**: LDA (Gensim) and NMF (Scikit-learn) for topic discovery
- **Preprocessing**: Advanced pipeline with NLTK/spaCy integration
- **Visualization**: Word clouds, pyLDAvis, and interactive topic exploration
- **Evaluation**: Coherence scores and comprehensive model comparison

## 🚀 Quick Start

1. **Setup Environment**:
   ```bash
   pip install -r requirements.txt
   python setup.py  # Download NLTK data and spaCy models
   ```

2. **Validate Dataset**:
   ```bash
   python simple_demo.py
   ```

3. **Run Complete Analysis**:
   ```bash
   python run_real_analysis.py
   ```

4. **Explore Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/topic_modeling_analysis.ipynb
   ```

## 📁 Project Structure

```
├── data/                  # BBC News dataset (place bbc_news.csv here)
├── src/                   # Core topic modeling modules
│   ├── data_loader.py     # Dataset loading utilities
│   ├── text_preprocessor.py # Advanced text preprocessing
│   ├── topic_modeling.py  # LDA and NMF implementations
│   ├── visualizations.py # Visualization utilities
│   └── config.py         # Configuration settings
├── notebooks/             # Jupyter analysis notebooks
├── results/              # Model outputs and processed data
├── visualizations/       # Generated plots and word clouds
├── simple_demo.py        # Quick dataset validation
├── run_real_analysis.py  # Complete analysis pipeline
└── requirements.txt      # Python dependencies
```

## 🛠️ Core Features

### Advanced Text Preprocessing
- **Tokenization & Lemmatization**: Using NLTK and spaCy
- **Stopword Removal**: Custom news-specific stopwords
- **Frequency Filtering**: Document frequency-based filtering
- **POS Tagging**: Keep meaningful parts of speech only

### Topic Modeling Algorithms
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling
- **NMF (Non-negative Matrix Factorization)**: Matrix factorization approach
- **Model Comparison**: Coherence scores and topic overlap analysis
- **Hyperparameter Tuning**: Configurable parameters for optimization

### Rich Visualizations
- **Word Clouds**: Visual representation of topic keywords
- **pyLDAvis**: Interactive topic exploration
- **Topic-Document Heatmaps**: Topic distribution analysis
- **Category Relationship Analysis**: Topics vs news categories

## 📊 Dataset Information

- **Source**: BBC News Dataset from Kaggle
- **Size**: 42,000+ articles
- **Categories**: Automatically extracted from URLs (business, sport, technology, etc.)
- **Content**: Combined titles and descriptions for rich text analysis
- **Time Period**: Recent BBC news articles with publication dates

## 🏆 Project Highlights

✅ **Professional Implementation**: Modular, reusable code  
✅ **Real Data Integration**: Works with actual BBC News dataset  
✅ **Comprehensive Analysis**: End-to-end topic modeling pipeline  
✅ **Rich Visualizations**: Multiple ways to explore topics  
✅ **Model Comparison**: LDA vs NMF performance analysis  
✅ **Ready for Production**: Saved models for deployment" 
