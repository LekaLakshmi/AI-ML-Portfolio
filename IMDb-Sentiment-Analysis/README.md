# IMDb Movie Review Sentiment Analysis

A beginner natural language processing project that classifies IMDb movie reviews as **positive** or **negative** using TF-IDF features and logistic regression.

## Project overview

The notebook loads the IMDb 50K review dataset, cleans the review text, splits the data into training and test sets, converts text to TF-IDF features, trains a logistic regression classifier, and reports accuracy, precision, recall, and F1 score.

## Dataset

The notebook expects a file named `IMDB Dataset.csv` in the same folder as the notebook. The CSV should contain these columns:

- `review`: the movie review text
- `sentiment`: `positive` or `negative`

Download the dataset from [IMDb Dataset of 50K Movie Reviews on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). The dataset is not included in this repository. Place the CSV beside the notebook before running it, and follow the dataset provider's terms for use and redistribution.

## Run the notebook

1. Install Python and Jupyter.
2. Install the packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. Put `IMDB Dataset.csv` in this project folder.
4. Open `imdb_sentiment_analysis.ipynb` in Jupyter and run the cells from top to bottom.

## Model and evaluation

- Text features: `TfidfVectorizer` with at most 5,000 features and no stop-word list (`stop_words=None`)
- Classifier: `LogisticRegression(max_iter=200)`
- Split: 80% training and 20% test, stratified by sentiment, with `random_state=42`
- Notebook result: 88.96% test accuracy; precision, recall, and F1 are approximately 0.89 for both classes

This is a baseline result from one train-test split, not a guarantee of performance on every dataset or split.
