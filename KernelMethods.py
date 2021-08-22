import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def DTWDistance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    if m == 0 and n == 0: # both sequences have length 0
        return 1
    elif m == 0 or n == 0: # only one of them has length 0
        return np.inf
    DTWMatrix = np.empty((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            DTWMatrix[i, j] = np.inf
    DTWMatrix[0, 0] = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            distance = np.abs(seq1[i-1] - seq2[j-1])
            DTWMatrix[i, j] = distance + min(DTWMatrix[i-1, j], DTWMatrix[i, j-1], DTWMatrix[i-1, j-1])
    return DTWMatrix[m, n]

def DTWDistanceMatrix(X):
    distanceMatrix = np.empty((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        print("Computing distance matrix. row", i + 1, "of", X.shape[0])
        for j in range(X.shape[0]):
            if i == j:
                distanceMatrix[i, j] = 0
            if i < j:
                distanceMatrix[i, j] = DTWDistance(X[i, :], X[j, :])
            else:
                distanceMatrix[i, j] = distanceMatrix[j, i]
    return distanceMatrix

def plotDTWDistancesMatrix(D):
    ax = sns.heatmap(D)
    plt.title("Distances between instances")
    plt.show()

def plotDTWSimilarityMatrix(S):
    ax = sns.heatmap(S)
    plt.title("Similarities between instances")
    plt.show()

def plotSortedDTWDistanceMatrix(originalD, y):
    D = originalD.copy() # in order to not change the original D matrix
    D = D[np.concatenate((np.where(y==-1), np.where(y==1)), axis=None),:]
    D = D[:,np.concatenate((np.where(y==-1), np.where(y==1)), axis=None)]
    plotDTWDistancesMatrix(D)

class DTWClassifier():
    def __init__(self, X_train, y_train, D, lbda=1, X_extended=None, D_extended=None):
        # We define the similarity between two instances as k(x, y) = exp(-c*d(x, y)), c = 1/max(D).
        # In this definition, d(x, y) is the DTW distance.
        self.c = 1 / np.max(D)
        self.K = np.exp(-self.c * D)
        # We use the dual representation of the kernel model for our prediction function:
        # f(x) = sum_i alpha_i k(x_i, x), where x_i are the training instances, and alpha = (K + lbda*I)^-1 * y
        self.X = X_train
        self.alpha = np.linalg.solve(self.K + lbda * np.eye(len(y_train)), np.eye(len(y_train))) @ y_train
        # For evaluation purposes, we provide the class with all available training data, as well as with the orginal
        # distances, so they don't have to be recomputed everytime we compare two known instances against another.
        self.X_extended = X_extended
        self.D_extended = D_extended
    def score(self, X):
        if X.ndim == 1:
            output = 0
            for i in range(len(self.alpha)):
                distance = DTWDistance(X, self.X[i, :])
                output[i] += self.alpha[i] * np.exp(-distance * self.c)
            return output
        elif X.ndim == 2:
            output = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                instanceIndex = -1
                if self.X_extended is not None:
                    for k in range(self.X_extended.shape[0]):
                        equality = True
                        for l in range(60):
                            if self.X_extended[k, l] != X[i, l]:
                                equality = False
                                break
                        if equality == True:
                            instanceIndex = k
                if instanceIndex >= 0:
                    for j in range(len(self.alpha)):
                        distance = self.D_extended[instanceIndex, j]
                        output[i] += self.alpha[j] * np.exp(-distance * self.c)
                else:
                    for j in range(len(self.alpha)):
                        distance = DTWDistance(X[i, :], self.X[j, :])
                        output[i] += self.alpha[j] * np.exp(-self.c * distance)
            return output
    def predict(self, X):
        scores = self.score(X)
        if X.ndim == 1:
            if scores <= 0:
                scores = -1
            else:
                scores = 1
        elif X.ndim == 2:
            scores[scores <= 0] = -1
            scores[scores > 0] = 1
        return scores
    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return sum(y_pred == y) / len(y)

# helper function to create k-fold train-test-splits
def create_kfold_mask(num_samples, k):
    masks = []
    fold_size = int(num_samples / k)
    for i in range(k):
        mask = np.zeros(num_samples, dtype=bool)
        mask[i * fold_size:(i + 1) * fold_size] = True
        masks.append(mask)
    return masks

def nestedCrossValidation(X, y, D, lbda_values=np.logspace(-4, 1, 241), k=9):
    print("Nested cross validation in progress: optimizing lambda for DTW Classification.")
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
                    clf_ij = DTWClassifier(X[train_indices],
                                           y[train_indices],
                                           D[np.ix_(train_indices, train_indices)],
                                           lbda,
                                           X,
                                           D[:, train_indices])
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
        clf_i =  DTWClassifier(X[tuning_indices],
                               y[tuning_indices],
                               D[np.ix_(tuning_indices, tuning_indices)],
                               best_lbda_i,
                               X,
                               D[:, tuning_indices])
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
    trained_tuned_and_tested = DTWClassifier(X, y, D, best_lbda, X, D)
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

class KernelPerceptron():
    def __init__(self, X, y, D):
        self.alpha = np.zeros(X.shape[1])
        self.X = X
        self.D = D
        self.c = 1/np.max(D)
        while True:
            alphaChanged = False
            for i in range(X.shape[0]):
                if y[i]*self.f(X[i]) <= 0:
                    self.alpha[i] += y[i]
                    alphaChanged = True
            if not alphaChanged:
                break
    def k(self, index_of_x1, x2):
        index2 = -1
        for i in range(self.X.shape[0]):
            match2 = True
            for l in range(len(self.X[i])):
                if self.X[i, l] != x2[l]:
                    match2 = False
                    break
            if match2: index2 = i
        if index2 >= 0:
            distance = self.D[index_of_x1, index2]
        else:
            distance = DTWDistance(self.X[index_of_x1], x2)
        return np.exp(-self.c * distance)
    def f(self, x):
        score = 0
        for i in range(len(self.alpha)):
            score += self.alpha[i] * self.k(i, x)
        if score <= 0:
            return -1
        else:
            return 1
    def predict(self, X):
        output = np.zeros(X.shape[0])
        for i in range(len(output)):
            output[i] = self.f(X[i])
        return output
    def accuracy(self, X_true, y_true):
        y_pred = self.predict(X_true)
        return sum(y_pred == y_true) / len(y_true)