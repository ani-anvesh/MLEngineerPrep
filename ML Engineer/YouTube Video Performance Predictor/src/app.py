from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('../models/youtube_view_predictor.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract all 7 features
        required_features = ['title_length', 'num_tags', 'category_id', 'publish_hour', 
                             'comments_disabled', 'ratings_disabled', 'video_error_or_removed']

        input_data = []
        for feature in required_features:
            value = data.get(feature)
            if value is None:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            input_data.append(value)

        features_array = np.array([input_data])
        prediction = model.predict(features_array)[0]

        return jsonify({'predicted_views': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
