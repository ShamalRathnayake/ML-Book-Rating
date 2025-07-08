# Book Recommendation System

## Overview

This project is a machine learning-based Book Recommendation System. It processes and analyzes book, user, and rating data to build a predictive model that estimates user ratings for books. The system leverages data preprocessing, feature engineering, and several regression models to predict how much a user will like a particular book.

## Features

- Data preprocessing and cleaning of book, user, and rating datasets
- Feature engineering (e.g., book popularity, user average rating, age group)
- Outlier detection and handling
- Model training using various regression algorithms (Random Forest, Gradient Boosting, Linear Regression, KNN)
- Model evaluation and selection
- Model serialization for later use (using `joblib`)
- Example prediction for new user-book pairs

## Project Structure

```
├── models.ipynb                # Main notebook for model training and evaluation
├── process_data.ipynb          # Data preprocessing and feature engineering
├── dist/
│   ├── book_rating_model.pkl   # Trained model (serialized)
│   └── rating_model_training_data.csv # Processed data used for training
├── .gitignore
```

## Data

- The project expects raw data files (books, users, ratings) in a `data/` directory (ignored by git).
- Preprocessed data is saved as `Preprocessed_Books_Data.csv`.

## Dependencies

Install the following Python packages (Python 3.7+ recommended):

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- missingno

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib missingno
```

## Usage

1. **Data Preprocessing**
   - Run `process_data.ipynb` to clean and merge the raw datasets. This will output a preprocessed CSV file.
2. **Model Training**
   - Run `models.ipynb` to perform feature engineering, train regression models, evaluate their performance, and save the best model to `dist/book_rating_model.pkl`.
3. **Prediction**
   - Load the trained model using `joblib` and provide new user-book data in the required format to predict ratings.

## Example: Predicting a Rating

```python
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('dist/book_rating_model.pkl')

# Example input data
new_data = pd.DataFrame({
    'user_id': [7589351],
    'isbn': [374157065],
    'book_title': ['Some Book'],
    'book_author': ['Author Name'],
    'publisher': ['Publisher XYZ'],
    'Language': ['en'],
    'Category': ['Fiction'],
    'age_group': ['Adult'],
    'age': [31],
    'year_of_publication': [2020],
    'book_popularity': [9],
    'user_avg_rating': [9],
    'book_avg_rating': [9],
})

# Predict
predicted_rating = model.predict(new_data)[0]
predicted_rating = np.clip(predicted_rating, 1, 10)
print(f'Predicted Rating: {predicted_rating}')
```

## License

This project is for educational and research purposes.
