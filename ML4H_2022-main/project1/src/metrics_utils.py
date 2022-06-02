from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def compute_metrics(labels: np.ndarray, prediction: np.ndarray, name: str):
    """
    prediction: softmax probabilities
    """
    n_classes = len(np.unique(labels))
    predicted_class = get_predicted_labels_from_probab(prediction)

    acc = metrics.accuracy_score(labels, predicted_class)
    f1 = metrics.f1_score(labels, predicted_class, average="macro")

    print(f"Test Accuracy Score: {acc}")
    print(f"Test Macro F1 Score: {f1}\n\n")

    if n_classes == 2:
        plot_roc_curve(labels, prediction, name)
        plot_pr_curve(labels, prediction, name)


def skorch_f1_score(net: Any, X: np.ndarray, y: np.ndarray) -> float:
    y_proba = net.predict_proba(X)
    return metrics.f1_score(get_predicted_labels_from_probab(y_proba), y, average="macro")


def sklearn_f1_score():
    return metrics.make_scorer(metrics.f1_score, average="macro")


def get_predicted_labels_from_probab(pred_probab: np.ndarray) -> np.ndarray:


    return pred_probab.argmax(axis=1)


def plot_roc_curve(labels: np.ndarray, prediction: np.ndarray, name: str):
    roc_auc = metrics.roc_auc_score(labels, prediction[:, 1])
    print(f"ROC AUC score {roc_auc}")
    fpr, tpr, _ = metrics.roc_curve(labels, prediction[:, 1])
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name=name)
    display.plot()

    plt.show()

def plot_pr_curve(labels: np.ndarray, prediction: np.ndarray, name: str):
    ap = metrics.average_precision_score(labels, prediction[:, 1])
    print(f"PR auc score {ap}")

    display = metrics.PrecisionRecallDisplay.from_predictions(labels, prediction[:, 1], name=name)
    _ = display.ax_.set_title("2-class Precision-Recall curve")

def plot_roc_curve2(labels: np.ndarray, prediction: np.ndarray, name: str):
    roc_auc = metrics.roc_auc_score(labels, prediction)
    print(f"ROC AUC score {roc_auc}")
    fpr, tpr, _ = metrics.roc_curve(labels, prediction)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    display.plot()
    plt.show()

def plot_pr_curve2(labels: np.ndarray, prediction: np.ndarray, name: str):
    ap = metrics.average_precision_score(labels, prediction)
    print(f"PR auc score {ap}")

    display = metrics.PrecisionRecallDisplay.from_predictions(labels, prediction, name=name)
    _ = display.ax_.set_title("2-class Precision-Recall curve")

def compute_metrics_from_keras(labels: np.ndarray, prediction: np.ndarray, name: str):
    
    if np.shape(prediction)[1] == 1:
        predicted_class = (prediction>0.5).astype(np.int8)
    else:
        predicted_class = get_predicted_labels_from_probab(prediction)

    acc = metrics.accuracy_score(labels, predicted_class)
    f1 = metrics.f1_score(labels, predicted_class, average="macro")

    print(f"Test Accuracy Score: {acc}")
    print(f"Test Macro F1 Score: {f1}\n\n")

    if np.shape(prediction)[1] == 1:
        plot_roc_curve2(labels, predicted_class, name)
        plot_pr_curve2(labels, predicted_class, name)
        

