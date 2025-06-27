import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def confusion_matrix_plot(y_test, y_pred):
    # Set the style of seaborn
    sns.set(style='whitegrid')

    # Create a confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['sad', 'joy', 'love', 'anger', 'fear'],
                yticklabels=['sad', 'joy', 'love', 'anger', 'fear'])
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.show()
