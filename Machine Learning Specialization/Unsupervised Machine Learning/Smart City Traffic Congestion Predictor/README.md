# Smart City Traffic Congestion Predictor

This project builds and trains deep learning models to predict traffic congestion levels (low, medium, high) using public traffic datasets such as METR-LA. The model leverages spatial-temporal data from loop detectors to classify congestion and help smart city applications.

---

## Project Structure

project/
├── data/ # Raw and processed datasets (METR-LA, PEMS-BAY)
│ ├── METR-LA/
│ └── PEMS-BAY/
├── models/ # Trained model weights and saved models
├── notebooks/ # Jupyter notebooks for experiments and EDA
│ └── smart_city_traffic_congestion_predictor.ipynb
├── outputs/ # Evaluation results, plots, and reports
├── src/ # Source code (data processing, training, evaluation)
│ ├── data_pipeline.py
│ └── evaluation.py
├── tests/ # Unit and integration tests
│ └── test_evaluation.py
├── README.md # This file
└── requirements.txt # Python dependencies

---

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/smart-city-traffic-predictor.git
cd smart-city-traffic-predictor/project
```

## Create and activate a Python virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download the METR-LA dataset (metr-la.h5) and place it under data/.

### Generate train/val/test datasets in .npz format:

```bash
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY (optional)
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5
```

## Evaluation

### Run model evaluation and generate classification reports and confusion matrices using:

```bash
python tests/test_data_pipeline.py
```

## Example Classification Report

                precision    recall  f1-score   support

            High       0.74      0.94      0.83      1695
          Medium       0.92      0.67      0.78      1730
             Low       0.00      0.00      0.00         0

        accuracy                           0.81      3425
       macro avg       0.55      0.54      0.54      3425
    weighted avg       0.83      0.81      0.80      3425

Make sure your trained model and validation data are properly loaded inside the script.
