import numpy as np
from matplotlib import pyplot as plt

def getTpFpTnFn(model, X, y_true):
    y_pred = model.predict(X)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y_true[i] == 1:
                fn += 1
            else:
                tn += 1
    return tp, fp, tn, fn

def precisionAndRecall(model, X, y_true):
    tp, fp, tn, fn = getTpFpTnFn(model, X, y_true)
    if tp == 0:
        return 0, 0
    return tp/(tp+fp), tp/(tp+fn)

def precisionRecallCurve(model, X, y):
    formerTheta0 = model.getThetaNormalized()[0].copy()
    precisions, recalls = [], []
    for theta0 in np.arange(-5, 5, 0.01):
        model.thetaNormalized[0] = theta0
        precision, recall = precisionAndRecall(model, X, y)
        if precision != 0 or recall != 0:
            recalls.append(recall)
            precisions.append(precision)
    model.thetaNormalized[0] = formerTheta0 # don't alter the model
    precision, recall = precisionAndRecall(model, X, y)
    plt.plot(recalls, precisions)
    plt.scatter([recall], [precision], color="red")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision/recall curve")
    plt.legend(["Precision/recall curve, with different theta_0", "this model"])
    plt.show()

def getFprTpr(model, X, y_true):
    tp, fp, tn, fn = getTpFpTnFn(model, X, y_true)
    if fp+tn == 0:
        fpr = 0
    else:
        fpr = fp/(fp+tn)
    if tp+fn == 0:
        tpr = 0
    else:
        tpr = tp/(tp+fn)
    return fpr, tpr

def ROCCurve(model, X, y_true):
    formerTheta0 = model.getThetaNormalized()[0].copy()
    truePositiveRates, falsePositiveRates = [], []
    for theta0 in np.arange(-5, 5, 0.01):
        model.thetaNormalized[0] = theta0
        fpr, tpr = getFprTpr(model, X, y_true)
        falsePositiveRates.append(fpr)
        truePositiveRates.append(tpr)
    model.thetaNormalized[0] = formerTheta0  # don't alter the model
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.show()

# Print feature importances
def plotFeatureImportances(theta):
    plt.barh(np.arange(len(theta)), abs(theta))
    plt.title("Feature importances")
    plt.ylabel("Feature nr.")
    plt.xlabel("Absolute weight")
    plt.tight_layout()
    plt.show()