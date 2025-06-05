# 📁 Scripts Folder

This folder contains the main Python scripts for training, evaluating, and interacting with the machine learning model.

## 🔧 Scripts

### `train_model.py`
- Trains an XGBoost classifier using scaled features.
- Uses GridSearchCV for hyperparameter tuning.
- Saves the best model to `models/xgb_model.pkl`.

### `predict_cli.py`
- Command-line interface for users to input values and get predictions.
- Loads the trained model and displays predicted class (e.g., Down, Neutral, Up).

---

## ▶️ Usage

Run scripts from the root directory of the project:

```bash
python scripts/train_model.py
python scripts/predict_cli.py

