import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score
)
import numpy as np
import mlflow
import matplotlib.pyplot as plt

def auroc(y_true, y_pred):
    """
    Calculate the AUROC (Area Under the Receiver Operating Characteristic)
    for each label separately.

    Args:
        y_true (torch.Tensor or np.ndarray):
            Ground truth labels of shape (num_samples, num_classes).
        y_pred (torch.Tensor or np.ndarray):
            Predicted probabilities of shape (num_samples, num_classes).
    
    Returns:
        dict: A dictionary with label indices as keys and corresponding AUROC
        scores as values.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    num_labels = y_true.shape[1]
    auroc_scores = {}

    for label_idx in range(num_labels):
        try:
            # Compute AUROC for each label
            auroc = roc_auc_score(y_true[:, label_idx], y_pred[:, label_idx])
            auroc_scores[label_idx] = auroc
        except ValueError as e:
            print(f"AUROC calculation error for label {label_idx}: {e}")
            auroc_scores[label_idx] = None

    return auroc_scores

def auprc(y_true, y_pred):
    """
    Calculate the AUPRC (Area Under the Precision-Recall Curve) for
    each label separately.

    Args:
        y_true (torch.Tensor or np.ndarray):
            Ground truth labels of shape (num_samples, num_classes).
        y_pred (torch.Tensor or np.ndarray):
            Predicted probabilities of shape (num_samples, num_classes).
    
    Returns:
        dict: A dictionary with label indices as keys and
        corresponding AUPRC scores as values.
    """
    
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    num_labels = y_true.shape[1]
    auprc_scores = {}

    for label_idx in range(num_labels):
        try:
            precision, recall, _ = precision_recall_curve(y_true[:, label_idx], y_pred[:, label_idx])
            
            # Compute AUPRC for each label
            auprc = auc(recall, precision)
            auprc_scores[label_idx] = auprc
        except ValueError as e:
            print(f"AUPRC calculation error for label {label_idx}: {e}")
            auprc_scores[label_idx] = None

    return auprc_scores

def f1(y_true, y_pred):
    assert len(y_pred.shape) <= 2, "y_pred must be 1D or 2D array"
    if len(y_pred.shape) == 2:
        y_pred = y_pred.squeeze(1)
        y_true = y_true.squeeze(1)

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores[:-1])  # Exclude last threshold
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    return best_f1

def log_precision_recall_curves_to_mlflow(y_true, y_pred, epoch):

    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    num_labels = y_true.shape[1]
    precision_recall_curves = {}

    if num_labels == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        try:
            precision, recall, _ = precision_recall_curve(
                y_true[:, 0], y_pred[:, 0]
            )
            ax.plot(recall, precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'PR Curve for Label')
        except ValueError as e:
            return

        plt.tight_layout()
        mlflow.log_figure(fig, f"PR_ep{epoch}.png")
        return

    fig, axes = plt.subplots(
        1, num_labels, figsize=(12, 4), sharey=True
    )
    axes = axes.flatten()

    for label_idx in range(num_labels):
        try:
            precision, recall, _ = precision_recall_curve(
                y_true[:, label_idx], y_pred[:, label_idx]
            )
            precision_recall_curves[label_idx] = (precision, recall)
            axes[label_idx].plot(recall, precision)
            axes[label_idx].set_xlabel('Recall')
            axes[label_idx].set_ylabel('Precision')
            axes[label_idx].set_title(f'PR Curve for Label {label_idx}')
        except ValueError as e:
            precision_recall_curves[label_idx] = None

    plt.tight_layout()
    mlflow.log_figure(fig, f"PR_ep{epoch}.png")

if __name__ == "__main__":
    # Example usage
    y_true = np.array([[0, 0], [1, 0], [1, 0], [0, 0], [1, 1]])
    y_pred = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.1], [0.1, 0.1], [0.9, 0.1]])
    auprc = auprc(y_true, y_pred)
    print(f"auprc: {auprc}")
    auroc = auroc(y_true, y_pred)
    print(f"auroc: {auroc}")
    log_precision_recall_curves_to_mlflow(y_true, y_pred, 1)
