
# PaySim Fraud Detection Using Machine Learning, BFS and DFS

## Project Overview

This project demonstrates financial transaction fraud detection using the PaySim1 simulated mobile-money transaction dataset. It combines machine learning classification with graph traversal algorithms.

### Main Features
- Logistic Regression for fraud classification.
- Random Forest Classifier for fraud classification.
- Data cleaning and exploratory data analysis.
- Confusion matrix and model evaluation metrics.
- Example transaction prediction.
- Breadth-First Search (BFS) and Depth-First Search (DFS).
- Graph visualization and downloadable output reports.

## Dataset

**Dataset:** PaySim1  
**Source:** https://www.kaggle.com/datasets/ealaxi/paysim1

The dataset contains simulated mobile-money transactions. The target column is `isFraud`:
- `0` — Non-fraud transaction
- `1` — Fraud transaction

The machine learning model uses transaction step, type, amount, and origin/destination balance features.

## BFS and DFS Implementation

The project samples up to **8,000 non-fraud transactions** from the cleaned PaySim dataset for the graph demonstration.

- Sender account IDs (`nameOrig`) represent source nodes.
- Receiver account IDs (`nameDest`) represent destination nodes.
- Transactions are represented as directed edges.
- BFS explores connected accounts level by level.
- DFS explores a path deeply before backtracking.

The BFS/DFS sample is separate from the ML training and testing workflow. These algorithms demonstrate graph traversal; they do not independently predict fraud.

## Machine Learning Workflow

1. Download the PaySim dataset.
2. Inspect and clean the data.
3. Explore transaction distributions and correlations.
4. Split the dataset into training and testing sets.
5. Preprocess numerical and categorical features.
6. Train Logistic Regression and Random Forest models.
7. Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and average precision.
8. Select the final baseline model using average precision.
9. Predict an example transaction.
10. Save model metrics, reports, visualizations, and graph traversal results.

## Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- NetworkX
- KaggleHub
- Joblib

## How to Run

1. Download or clone this repository.
2. Open `Fraud_Detection_PaySim_BFS_DFS_Single_Cell.ipynb` in Google Colab.
3. Select **Runtime → Run all**.
4. Wait for the dataset download, preprocessing, model training, evaluation, and graph traversal to complete.
5. Download the generated output ZIP when prompted.

An internet connection is required to download the dataset. Training on the full dataset may require considerable time and memory.

## Output Files

The notebook saves results in `/content/fraud_detection_outputs` and creates `/content/fraud_detection_outputs.zip`.

Generated outputs include:
- Fraud distribution charts
- Model comparison metrics
- Final model metrics
- Classification report
- Confusion matrix
- Random Forest feature importance
- Saved trained model
- BFS/DFS non-fraud sample
- BFS/DFS traversal results
- Transaction graph visualization
- Project results report

## Understanding the Confusion Matrix

- **True Positive (TP):** Fraud correctly predicted as fraud.
- **True Negative (TN):** Non-fraud correctly predicted as non-fraud.
- **False Positive (FP):** Non-fraud incorrectly predicted as fraud.
- **False Negative (FN):** Fraud incorrectly predicted as non-fraud.

Accuracy should not be considered alone because fraud datasets are often imbalanced. Precision, recall, F1-score, average precision, and false negatives are also important.

## Limitations

- PaySim is simulated data, not genuine bank transaction data.
- BFS and DFS demonstrate graph traversal and are not fraud prediction models.
- The graph demonstration uses only sampled non-fraud transactions.
- The project is an educational baseline and is not connected to real banking systems.

## Disclaimer

This project is for educational purposes only. Results on simulated data do not guarantee performance on real financial transactions.
