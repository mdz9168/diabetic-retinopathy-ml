from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def evaluate_model(y_test, y_pred):
    """
    this will evaluate model predictions using confusion matrix and classification report.

    Parameters:
    - y_test: true labels
    - y_pred: predicted labels

    Returns:
    - cm: confusion matrix
    - report: classification report as a dict
    """
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return cm, report

def save_report(report, filename):
    """
    this saves the classification report as a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(report, f, indent=4)

def plot_confusion_matrix(cm, labels, title, save_path):
    """
    this will save a heatmap of the confusion matrix.

    Parameters:
    - cm: confusion matrix
    - labels: class labels
    - title: plot title
    - save_path: path to save image
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
