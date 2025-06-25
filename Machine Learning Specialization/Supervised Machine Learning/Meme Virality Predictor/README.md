## 🎯 Project: Meme Virality Predictor

### 📌 Problem
Predict whether a meme will be low, medium, or highly viral based on text, time, and metadata.

### 📊 Features
- Sentiment score (from VADER)
- Caption length bucket
- Time of day
- Upvotes (log-transformed)
- Number of comments (log-transformed)
- Category

### 🛠️ Preprocessing
- Handled categorical and numeric features via ColumnTransformer
- Applied OneHotEncoding, StandardScaler
- Added Polynomial interaction features

### 🤖 Model
- Logistic Regression (Multinomial, LBFGS)
- Optional PolynomialFeatures for interactions

### 📈 Results

               precision    recall  f1-score   support

         high       0.99      0.98      0.98       264
          low       1.00      0.99      1.00       353
       medium       0.98      0.99      0.98       348
  
     accuracy                           0.99       965
    macro avg       0.99      0.99      0.99       965
 weighted avg       0.99      0.99      0.99       965

### 📎 Extras
- Used stratified train/test split
- Used VADER for sentiment
- Optional SMOTE for class balance (not required here)
