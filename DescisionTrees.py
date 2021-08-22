import numpy as np
from matplotlib import pyplot as plt

def information(probability):
    return -np.log2(probability)

def empiricalEntropy(observations):
    entropy = 0
    values, counts = np.unique(observations, return_counts=True)
    valueCounts = dict(zip(values, counts))
    for value in valueCounts:
        p = valueCounts[value] / len(observations)
        entropy += p*information(p)
    return entropy

def informationGain(x, y):
    output = empiricalEntropy(y)
    values, counts = np.unique(x, return_counts=True)
    valueCountsX = dict(zip(values, counts))
    for value in valueCountsX:
        freq = valueCountsX[value] / len(x)
        output -= freq*empiricalEntropy(y[x == value])
    return output

def informationGainRatio(x, y):
    return informationGain(x, y) / empiricalEntropy(x)

def continuousInformationGain(x, y, threshold):
    xSmallerEq = x[x <= threshold]
    ySmallerEq = y[x <= threshold]
    xGreater = x[x > threshold]
    yGreater = y[x > threshold]
    return empiricalEntropy(y) - len(xSmallerEq)/len(x)*empiricalEntropy(ySmallerEq) - len(xGreater)/len(x)*empiricalEntropy(yGreater)

class testNode():
    def __init__(self, criticalFeature, threshold, nodeSmallerEq, nodeGreater):
        self.criticalFeature = criticalFeature
        self.thr = threshold
        self.nodeSmallerEq = nodeSmallerEq
        self.nodeGreater = nodeGreater
    def predict(self, X):
        if X.ndim == 1:
            if X[self.criticalFeature] <= self.thr:
                return self.nodeSmallerEq.predict(X)
            else:
                return self.nodeGreater.predict(X)
        else:
            y_pred = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                y_pred[i] = self.predict(X[i,:])
            return y_pred
    def numberChildren(self):
        return self.nodeSmallerEq.numberChildren() + self.nodeGreater.numberChildren() + 2
    def getClassesInTrainingSet(self):
        return np.concatenate((self.nodeSmallerEq.getClassesInTrainingSet(), self.nodeGreater.getClassesInTrainingSet()), axis=None)
    def prune(self, tao):
        self.nodeSmallerEq.prune(tao)
        if len(self.nodeSmallerEq.getClassesInTrainingSet()) < tao and self.nodeSmallerEq.numberChildren() > 0:
            self.nodeSmallerEq = leafNode(self.nodeSmallerEq.getClassesInTrainingSet())
        self.nodeGreater.prune(tao)
        if len(self.nodeGreater.getClassesInTrainingSet()) < tao and self.nodeGreater.numberChildren() > 0:
            self.nodeGreater = leafNode(self.nodeGreater.getClassesInTrainingSet())
    def printCriticalAttributes(self):
        print("Critical attribute:", self.criticalFeature)
        self.nodeSmallerEq.printCriticalAttributes()
        self.nodeGreater.printCriticalAttributes()

class leafNode():
    def __init__(self, classesInTrainingSet):
        self.classesInTrainingSet = classesInTrainingSet
    def predict(self, x):
        classes, counts = np.unique(self.classesInTrainingSet, return_counts=True)
        indexOfMostFrequentClass = np.argmax(counts)
        return classes[indexOfMostFrequentClass]
    def numberChildren(self):
        return 0
    def getClassesInTrainingSet(self):
        return self.classesInTrainingSet
    def prune(self, tao):
        return None
    def printCriticalAttributes(self):
        return None

def C45(X, y):
    if len(set(y)) == 1:
        return leafNode(y)
    else:
        bestJ = 0
        bestThreshold = 0
        bestInformationGain = -np.inf
        for j in range(X.shape[1]):
            x = X[:,j]
            for threshold in x:
                newInformationGain = continuousInformationGain(x, y, threshold)
                if newInformationGain > bestInformationGain:
                    bestInformationGain = newInformationGain
                    bestJ = j
                    bestThreshold = threshold
        nodeSmallerEq = C45(X[X[:,bestJ] <= bestThreshold,:], y[X[:, bestJ] <= bestThreshold])
        nodeGreater = C45(X[X[:,bestJ] > bestThreshold,:], y[X[:, bestJ] > bestThreshold])
        return testNode(bestJ, bestThreshold, nodeSmallerEq, nodeGreater)

# helper function to create k-fold train-test-splits
def create_kfold_mask(num_samples, k):
    masks = []
    fold_size = int(num_samples / k)
    for i in range(k):
        mask = np.zeros(num_samples, dtype=bool)
        mask[i * fold_size:(i + 1) * fold_size] = True
        masks.append(mask)
    return masks

def nestedCrossValidationRandomForest(X, y, lbda_values=np.logspace(1, 3, 5), numberOfInstances=100, proportionOfAttributes=0.25, k=9):
    print("Nested cross validation in progress: optimizing number of trees for random forest.")
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
            print(" lbda =", lbda)
            risks_i_lbda = np.zeros(k)
            for j in range(k):
                print("  j =", j+1, "of", k)
                if j != i:
                    train_indices = masks[i] == False
                    train_indices[masks[j]] = False
                    # Train model
                    forest_ij = randomForest(X[train_indices], y[train_indices], lbda, numberOfInstances, proportionOfAttributes)
                    # Determine risk on S_j
                    y_pred_ij = forest_ij.predict(X[masks[j]])
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
        forest_i = randomForest(X[tuning_indices], y[tuning_indices], best_lbda_i, numberOfInstances, proportionOfAttributes)
        # Determine risk for i on S_i
        y_pred_i = forest_i.predict(X[masks[i]])
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
    trained_tuned_and_tested = randomForest(X, y, best_lbda, numberOfInstances, proportionOfAttributes)
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

class randomForest():
    def __init__(self, X, y, numberOfTrees, numberOfInstances, portionOfAttributes):
        numberOfTrees = int(numberOfTrees)
        self.trees = np.empty(numberOfTrees, dtype=leafNode)
        numberOfAttributes = (np.ceil(portionOfAttributes*X.shape[1])).astype(int)
        self.attributes = np.empty((numberOfTrees, numberOfAttributes), dtype=int)
        for i in range(numberOfTrees):
            trainingInstances = np.random.randint(X.shape[0], size=numberOfInstances)
            trainingAttributes = np.random.choice(X.shape[1], numberOfAttributes, replace=False)
            X_train = X[trainingInstances, :]
            X_train = X_train[:, trainingAttributes]
            y_train = y[trainingInstances]
            self.trees[i] = C45(X_train, y_train)
            self.attributes[i,:] = trainingAttributes
    def predict(self, X):
        votes = np.empty(X.shape[0])
        for i in range(len(self.trees)):
            votes += self.trees[i].predict(X[:,self.attributes[i,:]])
        votes[votes <= 0] = -1
        votes[votes > 0] = 1
        return votes
    def accuracy(self, X, y_true):
        y_pred = self.predict(X)
        return sum(y_pred == y_true) / len(y_pred)