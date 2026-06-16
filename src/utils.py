from sklearn.metrics import roc_curve , roc_auc_score , precision_recall_curve ,average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, threshold=0.5, save=True):
    y_pred_binary = (y_pred >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save, dpi=300, bbox_inches='tight')
    plt.show()
    

def print_classification_metrics(y_true, y_pred, threshold=0.5):
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_binary):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred_binary):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred_binary):.4f}")


def plot_pr_curve_with_ap(y_true, y_pred, label=None, save=True):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    ap_score = average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    plt.plot(recalls, precisions, linewidth=2, label=f'{label} (AP = {ap_score:.3f})' if label else f'AP = {ap_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()
    return ap_score


def plot_roc_curve_with_auc(y_true, y_pred, label=None, save=True):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {auc_score:.3f})' if label else f'AUC = {auc_score:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()
    return auc_score

def plot_roc_curve(y_true , y_pred , label=None, save=True):
    fpr, tpr, thresholds = roc_curve(y_true , y_pred)

    plt.figure(figsize=(18,5))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="upper left")
    plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()


def plot_precision_vs_recall(y_true, y_pred , save=True):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    plt.figure(figsize=(18,7))
    plt.plot(recalls[:-1], precisions[:-1], "b-", label="Precision")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.savefig(save, dpi=300, bbox_inches='tight')

    plt.show()


