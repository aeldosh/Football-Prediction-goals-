# Football Player Performance Predictor ⚽

## Overview
This project is a final year Data Science Laboratory project that predicts the number of goals a football player will score based on their performance statistics. The application features a robust machine learning backend powered by a **Gradient Boosting Regressor**, trained on the **FBRef 2024-2025 dataset**. The model achieved optimal performance using 10-Fold Cross Validation. 

A sleek and professional **Streamlit** web application serves as the frontend, allowing users to input player metrics and get instant predictions.

## Features
- **Accurate Predictions**: Uses a fine-tuned Gradient Boosting Regressor to estimate a player's goals.
- **Comprehensive Feature Set**: Analyzes age, matches played, minutes, assists, penalty goals, yellow/red cards, and progressive carries/passes.
- **Beautiful UI**: Modern, glassmorphic, and dynamic user interface built with custom CSS in Streamlit.
- **Model Evaluation**: Employs rigorous 10-fold cross-validation, comparing multiple algorithms including Linear Regression, Decision Trees, Random Forest, and Support Vector Regression.

## Project Structure
- `app.py`: The main Streamlit web application script.
- `train_and_save_model.py`: The script used to preprocess data, train multiple models, evaluate them, and save the best one.
- `best_model.pkl`: The serialized Gradient Boosting Regressor model.
- `PlayersFBREF_FeatureSelected.csv`: The processed dataset used for model training.
- `requirements.txt`: List of dependencies required to run the project.
- Word Documents: Supplementary project documentation for the Data Science Laboratory.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-github-repo-url>
cd Football_Player_Predictor
```

### 2. Install dependencies
Ensure you have Python 3.8+ installed. Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### 3. Run the application
Start the Streamlit app locally:
```bash
streamlit run app.py
```

## Model Training
If you wish to re-train the model, run the training script:
```bash
python train_and_save_model.py
```
This will output the evaluation metrics for all tested algorithms and save a new `best_model.pkl` to the directory.

## License
This project was created for educational purposes as a Data Science Laboratory Final Year Project.
