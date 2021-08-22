import numpy as np
from matplotlib import pyplot as plt

def L2Regularizer(theta):
    return sum(theta*theta)

# compute weight vector theta by ridge classification
def ridgeClassification(X, y, lbda=1):
    theta = np.linalg.solve(np.transpose(X) @ X + lbda*np.eye(X.shape[1]), np.eye(X.shape[1])) @ np.transpose(X) @ y
    return theta

# A ridge classifier is basically a ridge regressor, that outputs 1 if the predicted value is >1, and -1 otherwise.
class RidgeClassifier():
    def __init__(self, X, y, lbda=1):
        X_train = X.copy()
        self.means = np.zeros(X.shape[1])
        self.stds = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            self.means[j] = X[:,j].mean()
            self.stds[j] = X[:, j].std()
            X_train[:,j] = (X_train[:,j] - self.means[j]) / self.stds[j]
        self.thetaNormalized = ridgeClassification(np.hstack((np.ones((X_train.shape[0], 1)), X_train)), y, lbda)
        self.theta = ridgeClassification(np.hstack((np.ones((X.shape[0], 1)), X)), y, lbda)
    def score(self, X):
        X_query = X.copy()
        for j in range(X_query.shape[1]):
            X_query[:,j] = (X_query[:,j] - self.means[j]) / self.stds[j]
        return X_query @ self.thetaNormalized[1:len(self.thetaNormalized)] + self.thetaNormalized[0]
    def predict(self, X):
        scores = self.score(X)
        scores[scores <= 0] = -1
        scores[scores > 0] = 1
        return scores
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return sum(y_pred == y_true) / len(y_true)
    def getThetaNormalized(self):
        return self.thetaNormalized

# In our case, this is identical to the 0/1 loss, but we define it for completeness.
def perceptronLoss(y_pred, y_true):
    return max(0, -y_pred*y_true)

class Perceptron():
    def __init__(self, X, y):
        n = X.shape[0]
        X_train = X.copy()
        self.means = np.zeros(X.shape[1])
        self.stds = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            self.means[j] = X[:, j].mean()
            self.stds[j] = X[:, j].std()
            X_train[:, j] = (X_train[:, j] - self.means[j]) / self.stds[j]
        X_train = np.hstack((np.ones((n, 1)), X))
        self.thetaNormalized = np.ones(X_train.shape[1])
        while True:
            thetaChanged = False
            for i in range(n):
                y_pred = np.dot(X_train[i], self.thetaNormalized)
                if perceptronLoss(y_pred, y[i]) > 0:
                    self.thetaNormalized += y[i] * X_train[i]
                    thetaChanged = True
            if not thetaChanged:
                break
    def score(self, X):
        X_query = X.copy()
        for j in range(X_query.shape[1]):
            X_query[:, j] = (X_query[:, j] - self.means[j]) / self.stds[j]
        return X_query @ self.thetaNormalized[1:len(self.thetaNormalized)] + self.thetaNormalized[0]
    def predict(self, X):
        scores = self.score(X)
        scores[scores <= 0] = -1
        scores[scores > 0] = 1
        return scores
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return sum(y_pred == y_true) / len(y_true)

# helper function to create k-fold train-test-splits
def create_kfold_mask(num_samples, k):
    masks = []
    fold_size = int(num_samples / k)
    for i in range(k):
        mask = np.zeros(num_samples, dtype=bool)
        mask[i * fold_size:(i + 1) * fold_size] = True
        masks.append(mask)
    return masks

def nestedCrossValidationRidgeClassification(X, y, lbda_values=np.logspace(-3, 2, 151), k=18):
    print("Nested cross validation in progress: optimizing lambda for Ridge Classification.")
    masks = create_kfold_mask(X.shape[0], k)
    risk = np.zeros(k)
    risk_per_lbda = dict.fromkeys(lbda_values)
    for lbda in lbda_values:
        risk_per_lbda[lbda] = 0
    for i in range(k):
        print("i =", i+1, "of", k)
        min_risk_lbda = np.inf
        best_lbda_i = 0
        for lbda in lbda_values:
            # print(" lbda =", lbda)
            risks_i_lbda = np.zeros(k)
            for j in range(k):
                if j != i:
                    train_indices = masks[i] == False
                    train_indices[masks[j]] = False
                    # Train model
                    clf_ij = ridgeClassification(X[train_indices], y[train_indices], lbda)
                    # Determine risk on S_j
                    y_pred_ij = clf_ij.predict(X[masks[j]])
                    risks_i_lbda[j] = sum(y_pred_ij != y[masks[j]]) / len(y_pred_ij)
                    risk_per_lbda[lbda] += risks_i_lbda[j]
            # Average R_S_j to determine risk...
            avg_risk_i_lbda = sum(risks_i_lbda) / (k-1)
            # Choose lambda that minimizes risk
            if avg_risk_i_lbda <= min_risk_lbda:
                best_lbda_i = lbda
                min_risk_lbda = avg_risk_i_lbda
        # train model on S \ S_i
        tuning_indices = masks[i] == False
        clf_i =  ridgeClassification(X[tuning_indices], y[tuning_indices], best_lbda_i)
        # Determine risk for i on S_i
        y_pred_i = clf_i.predict(X[masks[i]])
        risk[i] = sum(y_pred_i != y[masks[i]]) / len(y_pred_i)
        # store best lbda
        print(" Best Lambda:", best_lbda_i)
    # average risks
    avg_risk = risk.mean()
    # determine best lambda
    lowest_lambda_risk = np.inf
    best_lbda = 0
    for lbda in lbda_values:
        if risk_per_lbda[lbda] <= lowest_lambda_risk:
            best_lbda = lbda
            lowest_lambda_risk = risk_per_lbda[lbda]
    # train model on all data
    trained_tuned_and_tested = ridgeClassification(X, y, best_lbda)
    # Plot the relevant data
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    lbdaRisksVector = []
    for lbda in lbda_values:
        lbdaRisksVector.append(risk_per_lbda[lbda] / (k * (k - 1)))
    # print("lbdaRisksVector:", lbdaRisksVector)
    ax.plot(lbda_values, lbdaRisksVector)
    plt.xlabel("lambda")
    plt.ylabel("averaged risk over all S_i, S_j")
    ax.set_xscale("log")
    plt.title("Nested Cross Validation: average risk per lambda")
    plt.show()
    return trained_tuned_and_tested, best_lbda, avg_risk