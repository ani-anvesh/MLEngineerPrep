# 🌍 Tourist Destination Recommender

A content-based travel recommendation system that suggests global tourist destinations based on user preferences like culture, history, beach, and more. Built using TF-IDF and cosine similarity, the system ranks destinations by how well their features match the user's interests, with optional scoring weights for popularity and best travel times.

---

## 📌 Approach

This recommender uses a **content-based filtering** approach:

1. **TF-IDF Vectorization**  
   - Converts the `Category` column (e.g., "beaches, museums, nightlife") into numerical vectors.

2. **Cosine Similarity**  
   - Compares the user input vector with each destination vector to compute similarity scores.

3. **Weighted Scoring**  
   - Combines similarity and popularity:

```python final_score = 0.8 * similarity_score + 0.2 * popularity_score ```

## Optional Filtering

Filters destinations by travel month (e.g., "May") using the Best_Time_to_Travel column.

## 🧠 Scoring Logic
Component	Description
Similarity Score	Cosine similarity between user preferences and destination
Popularity Score	Normalized score (0–1) representing destination popularity
Final Score	Weighted combination of both scores for final ranking

## 🎯 User Input Format
Themes (required): comma-separated string
Example: "culture, history, museums"

Month (optional): a single month string
Example: "May"

## ✅ Sample Output
Input:

makefile
Always show details

Copy
Preferences: culture, history, museums  
Month: May
Top 5 Recommendations:

Rank	City	Country	Similarity	Final Score
1	Berlin	Germany	0.5766	0.6558
2	Madrid	Spain	0.5689	0.6478
3	Amsterdam	Netherlands	0.4769	0.5706
4	Paris	France	0.4095	0.5258
5	Stockholm	Sweden	0.4013	0.4956

## 📊 Visualization
A bar chart is generated to show similarity scores for the top 5 recommended destinations. The chart is saved to:

bash
Always show details

Copy
outputs/top_5_recommendations.png
## 🧰 Tech Stack
Python 🐍

Pandas & NumPy

Scikit-learn (TF-IDF, cosine similarity)

Matplotlib (visualization)

## 📂 Project Structure
bash
Always show details

Copy
project/
├── data/                     # Raw and cleaned data files
├── models/                   # Serialized models (if any)
├── notebooks/                # Jupyter notebooks for exploration and testing
├── outputs/                  # Final recommendations, charts
├── src/
│   ├── data_pipeline.py      # Data loading and cleaning
│   ├── evaluation.py         # Scoring functions
│   └── recommender.py        # Recommendation logic
├── tests/                    # Unit tests
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies

## 📥 Getting Started
Clone this repo:

bash
Always show details

Copy
git clone <repo-url>
cd project
Install dependencies:

bash
Always show details

Copy
pip install -r requirements.txt
Run the notebook in notebooks/ to test with your own inputs!

✨ Future Enhancements
Add collaborative filtering

Integrate map visualizations

Deploy as a web app (e.g., Streamlit or Flask)

Include real-time destination data via API

##  📬 Contact
Feel free to reach out for collaboration or suggestions!
"""

Save to file
readme_path = Path("/mnt/data/README.md")
readme_path.write_text(readme_content)

readme_path.name # Return file name only