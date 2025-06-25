# Music Mood Predictor

## Project Overview
This project builds a **Music Mood Predictor** using a logistic regression model to classify songs into two moods: **Happy** or **Sad**.  
The goal is to predict the mood of a song based on audio features such as valence, energy, danceability, and more.

---

## Dataset
- The dataset contains audio features extracted from songs, including:
  - `acousticness`, `danceability`, `duration_ms`, `energy`, `instrumentalness`, `key`, `liveness`, `loudness`, `mode`, `speechiness`, `tempo`, `time_signature`, `valence`
- Each song is labeled with a `target`:
  - 1 = Happy mood
  - 0 = Sad mood
- Additional metadata such as `song_title` and `artist` are included but not used for modeling.

---

## Exploratory Data Analysis (EDA)
- Visualized distributions of features with histograms.
- Investigated correlations between features and with the target using heatmaps.
- Used scatter plots and pair plots to understand feature relationships and separability by mood.
- Identified no missing values in the dataset.

---

## Data Preprocessing
- Dropped irrelevant columns (e.g., index columns).
- Split data into features (`X`) and target (`y`).
- Scaled features using `StandardScaler` to standardize the range for better model convergence.
- Split data into training and testing sets (e.g., 80% train, 20% test).

---

## Modeling
- Trained a **Logistic Regression** model using the training set.
- Logistic regression uses the sigmoid function to output probabilities for mood classes.
- Model learned coefficients representing the influence of each feature on the mood prediction.

---

## Evaluation
- Achieved an accuracy of approximately **65%** on the test set.
- Confusion matrix revealed balanced performance between classes:
  - True Positives (Happy): 129
  - True Negatives (Sad): 134
  - False Positives: 72
  - False Negatives: 69
- Precision, recall, and F1-scores around 0.64–0.65, indicating moderate predictive power.
- Visualized confusion matrix for better interpretability.

---

## Future Improvements
- Experiment with more complex models such as Random Forest or XGBoost to capture nonlinear relationships.
- Perform feature engineering and selection to improve model input quality.
- Explore hyperparameter tuning and cross-validation.
- Expand dataset size or include more diverse audio features.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ani-anvesh/MLEngineerPrep.git
