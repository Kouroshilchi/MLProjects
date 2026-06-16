from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

def plot_roc_curve(y_true , y_pred , label=None):
    fpr, tpr, thresholds = roc_curve(y_true , y_pred)

    plt.figure(figsize=(18,5))
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plot_roc_curve(fpr, tpr)
    plt.show()