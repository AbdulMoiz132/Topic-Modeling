# BBC News Dataset

This folder contains the BBC News dataset from Kaggle for topic modeling analysis.

## Dataset File

- **`bbc_news.csv`** - Main BBC News dataset from Kaggle

## Dataset Structure

The dataset contains BBC news articles with the following columns:
- **title**: Article headline
- **pubDate**: Publication date
- **guid**: Unique identifier
- **link**: BBC article URL (used for category extraction)
- **description**: Article summary/content

## Categories

Categories are automatically extracted from BBC URLs:
- **business**: Financial and economic news
- **sport**: Sports news and events
- **technology**: Tech and digital innovation
- **entertainment**: Arts, culture, and entertainment
- **politics**: Political news and government
- **health**: Health and medical news
- **science**: Scientific research and discoveries
- **world**: International news
- **uk**: UK domestic news
- **general**: Other news topics

## Data Processing

The dataset is processed by:
1. **Category Extraction**: Parsing categories from BBC URLs
2. **Text Combination**: Merging titles and descriptions
3. **Quality Filtering**: Removing very short or empty articles
4. **Preprocessing**: Tokenization, cleaning, and lemmatization

## Usage

Use the data loader in `src/data_loader.py` or run:
```python
python simple_demo.py  # Quick validation
python run_real_analysis.py  # Full analysis
```
