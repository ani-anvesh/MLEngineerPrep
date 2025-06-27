# 🖼️ Art Style Classifier

This project implements an image classification pipeline to classify artworks by their artistic styles using deep learning techniques. It consists of a data pipeline, model training, and evaluation scripts, along with accompanying unit tests.

---

## 🎨 Project — Art Style Classifier (4 hrs)

**Goal**: Classify paintings into styles (e.g., Impressionism, Cubism, Baroque) using image features and recommend similar works.

| Task                    | Duration | Output                                                                                                                                      |
| ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 📂 Dataset Prep         | 30 mins  | Use WikiArt Dataset or a smaller Kaggle image-style set                                                                                     |
| 🧹 Preprocessing        | 30 mins  | Resize images, normalize pixel values, optionally convert to grayscale                                                                      |
| 🧠 Model Training       | 90 mins  | Use CNN via Keras or PyTorch<br>Use pretrained models (e.g., MobileNetV2 or ResNet) for feature extraction<br>Train classifier on top layer |
| 📊 Evaluate + Recommend | 45 mins  | Evaluate top-1 accuracy<br>Use cosine similarity on extracted features to recommend similar paintings                                       |
| 📜 Document             | 25 mins  | Add screenshots and model analysis to README.md                                                                                             |

---

## 📁 Project Structure

.
├── art-style-classifier.ipynb # Jupyter notebook for training and evaluating the model
├── data_pipeline.py # Contains the data loading and transformation logic
├── evaluation.py # Model evaluation metrics and utilities
├── test_data_pipeline.py # Unit tests for the data pipeline

yaml
Always show details

Copy

---

## 🧪 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd art-style-classifier
   Install dependencies
   ```

bash
Always show details

Copy
pip install -r requirements.txt
Recommended Environment

Python 3.8+

PyTorch

torchvision

numpy

pandas

scikit-learn

matplotlib

🚀 Running the Project
Run the Jupyter Notebook
Open art-style-classifier.ipynb in JupyterLab or VS Code to train and evaluate the model interactively.

Use the data pipeline programmatically

python
Always show details

Copy
from data_pipeline import get_dataloaders

train_loader, val_loader = get_dataloaders(data_dir="path/to/data")
Evaluate Model

python
Always show details

Copy
from evaluation import evaluate_model
evaluate_model(model, val_loader)
✅ Testing
To run unit tests:

bash
Always show details

Copy
pytest test_data_pipeline.py
📊 Features
Image classification using CNNs (ResNet, etc.)

Custom data loaders with augmentations

Evaluation metrics: Accuracy, Precision, Recall, F1-score

Jupyter notebook-based training interface

Unit testing with pytest

📎 License
This project is open source and available under the MIT License.

✍️ Author
Anvesh Reddy
Feel free to reach out for collaborations or questions.
