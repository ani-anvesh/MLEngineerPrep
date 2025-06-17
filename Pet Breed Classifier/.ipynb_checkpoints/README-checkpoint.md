# Pet Breed Classifier

## Overview
A simple CNN to classify images of cats and dogs.

## Dataset
- 447 training images (cats: 224, dogs: 223)
- 110 validation images (cats: 55, dogs: 55)
- 140 test images (cats: 70, dogs: 70)

## Model Architecture
- 3 Conv2D + MaxPooling layers
- Flatten
- Dense layer (128 units)
- Output Dense layer with sigmoid for binary classification

## Training Procedure
- Binary crossentropy loss, Adam optimizer
- Early stopping with patience 5 on validation loss
- Model checkpoint saving best weights

## Results
- Training/Validation accuracy & loss curves
- Test set accuracy
- Confusion matrix

## Challenges & Future Work
- Data augmentation for better generalization
- Experiment with dropout to reduce overfitting
- Increase dataset size or use pre-trained models

## Results

### Evaluation Metrics on Test Set:
- **Accuracy**: 55.7%
- **Precision**: 55.8%
- **Recall**: 55.7%
- **F1 Score**: 55.6%
- **ROC AUC**: 58.3%

### Classification Report:

                   precision    recall  f1-score   support

             Cat       0.56      0.51      0.54        70  
             Dog       0.55      0.60      0.58        70  
        accuracy                           0.56       140  
       macro avg       0.56      0.56      0.56       140
    weighted avg       0.56      0.56      0.56       140


### Observations:
- The model is slightly better at detecting dogs than cats.
- ROC AUC of 0.58 indicates weak but better-than-random performance.
- Accuracy is low, suggesting more training or better augmentation/regularization is needed.