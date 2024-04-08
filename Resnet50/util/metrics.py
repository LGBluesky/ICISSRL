from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
import numpy as np


def evaluate2(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.empty((y_scores.shape[0]))

    threshold_confusion = 0.5
    for i in range(y_scores.shape[0]):
        if y_scores[i, 0] >= threshold_confusion:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores[:, 1])
    AUC_ROC = roc_auc_score(y_true, y_scores[:, 1])

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores[:, 1])
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))

    # Confusion matrix

    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))
    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))

    evaluate_state = {'accuracy': accuracy,
                'specificity': specificity,
                'sensitivity': sensitivity,
                'precision': precision,
                'F1_score': F1_score,
                'AUC_ROC': AUC_ROC}
    return evaluate_state, tpr, fpr, AUC_ROC

def show_ROC(tpr, fpr, AUC_ROC, key, root):
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    plt.plot(fpr, tpr, '-')
    plt.title('ROC curve, AUC = ' + str(AUC_ROC), fontsize=14)
    plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig(os.path.join(root, key+'_'+'ROC'+'.png'), dpi=300)