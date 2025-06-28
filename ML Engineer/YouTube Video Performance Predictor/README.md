
# 🎥 YouTube Video Performance Predictor API

This project is a **machine learning-powered Flask API** that predicts the number of views a YouTube video might get, based on its metadata (title length, tags, category, etc.).

---

## 📌 Project Structure

```
project/
├── data/                    # Raw dataset (USvideos.csv)
├── models/                  # Saved ML model (.pkl)
├── notebooks/               # Jupyter notebook for training
├── outputs/                 # Any generated outputs (plots, metrics)
├── src/                     # Source code: data pipeline, API, etc.
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## ✅ Dataset Used

- **Source:** [Kaggle - YouTube Trending Video Dataset](https://www.kaggle.com/datasets/datasnaek/youtube-new/data)  
- **File Used:** `USvideos.csv`  
- **Target Variable:** `views`  

---

## ✅ Features Used for Prediction

| Feature | Description |
|---|---|
| `title_length` | Number of characters in the video title |
| `num_tags` | Number of tags used |
| `category_id` | YouTube category ID |
| `publish_hour` | Hour of day the video was published |
| `comments_disabled` | 1 if comments are disabled, else 0 |
| `ratings_disabled` | 1 if likes/dislikes are disabled, else 0 |
| `video_error_or_removed` | 1 if video is removed or has error, else 0 |

---

## ✅ Model

- **Algorithm Used:** RandomForestRegressor  
- **Training:** Done inside Jupyter Notebook (`notebooks/youtube_video_performance_predictor.ipynb`)  
- **Saved Model:** `models/youtube_view_predictor.pkl`

---

## ✅ Setup Instructions

### 1. Clone and Navigate:

```bash
git clone <your-repo-url>
cd project
```

### 2. Create Virtual Environment:

```bash
python -m venv venv
source venv/bin/activate      # For Mac/Linux
# OR
venv\Scripts\activate       # For Windows
```

### 3. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run Flask API:

```bash
python src/app.py
```

API will run on:

```
http://127.0.0.1:5000
```

---

## ✅ API Usage

### 🎯 Endpoint:

```
POST /predict
```

### ✅ Example Input JSON:

```json
{
  "title_length": 50,
  "num_tags": 5,
  "category_id": 22,
  "publish_hour": 17,
  "comments_disabled": 0,
  "ratings_disabled": 0,
  "video_error_or_removed": 0
}
```

### ✅ Example cURL Request:

```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{
  "title_length": 50,
  "num_tags": 5,
  "category_id": 22,
  "publish_hour": 17,
  "comments_disabled": 0,
  "ratings_disabled": 0,
  "video_error_or_removed": 0
}'
```

### ✅ Example API Response:

```json
{
  "predicted_views": 123456
}
```

*(Note: Prediction number will vary based on your trained model.)*

---

## ✅ Testing

- Test the API using **Postman**, **curl**, or any API client.
- For unit testing the data pipeline, see:  
`tests/test_data_pipeline.py`

---

## ✅ Dependencies (requirements.txt)

```
Flask
numpy
pandas
scikit-learn
joblib
```

(Add others like XGBoost if used.)

---

## ✅ Future Improvements

- Deploy API to cloud (AWS/GCP/Heroku)
- Add more features (e.g., NLP on titles)
- Hyperparameter tuning
- Frontend for input/output visualization
