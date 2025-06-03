# 📓 Jupyter Notebooks Overview

This folder contains the Jupyter Notebook(s) used to execute the core stages of the stock analysis workflow. The notebooks are structured for interactive development, combining data preprocessing, analysis, and visualization in a clear, reproducible format.

## 📘 Notebooks Included

- **`01_cleaning_feature_engineering.ipynb`**  
  Loads raw datasets, performs cleaning and formatting, merges technical and statistical data, and generates key features such as moving averages, momentum indicators, and volatility labels. This notebook produces the final dataset used for visualization and modeling.

- **`02_data_visualization.ipynb`**  
  Explores stock performance trends using Matplotlib and Seaborn. Includes distribution plots, technical indicator overlays, and volatility-level breakdowns to uncover patterns and insights.

- **`03_increase_data_to_5yr.ipynb`**  
  Use the metrics from the last two data sets and python notebooks to increment the amount of data, in this case from the last 1.5 years to 5 years of stock data that will be mainly use for machine learning model.

- **`04_model_building.ipynb`**  
  Builds a machine learning model to classify or predict stock behavior based on engineered features. Includes training, testing, and evaluation using accuracy and performance metrics.

## 🛠 Tools Used

- Python (Jupyter)
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

> 🧠 These notebooks provide a transparent view into the project’s logic and progression, ideal for reviewing data transformations and insight generation.
