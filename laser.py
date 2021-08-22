import scipy.io
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

import DataAnalysis
import DescisionTrees
import LinearModels
import KernelMethods
import Evaluation

# Load the file and get X and y from it
matlabFile = scipy.io.loadmat('laser.mat')
X = matlabFile["X"]
y = -matlabFile["Y"].flatten() # "-1" means intact, "+1" means defect

########################################
### FIRST ANALYSIS AND PREPROCESSING ###
########################################

# First, let's have a look at the data and visualize all intensity curves.
DataAnalysis.plotIndividualIntensitySeries(X, y)
# There are two outliers (instances 35 and 162) that can be determined with np.where(X[:,59] > 30).
# Maybe with these two outliers, the intensity was measured to increase, instead of decrease. We could correct for that,
# but since we're not sure about that, we instead drop these two instances later, since they make up only one percent
# of the training data.

# Let's quick have a look at how well intact and defect lasers are separated.
# As a distance measure, we use the Dynamic Time Warp distance. We build the distance matrix and plot it.
D = KernelMethods.DTWDistanceMatrix(X)
KernelMethods.plotDTWDistancesMatrix(D)
# Here we clearly see the two outliers, since they have great distances to the other instances. Now might be a good time
# to get rid of them.
outliers = [35, 162]
X = np.delete(X, outliers, axis=0)
y = np.delete(y, outliers, axis=0)
D = np.delete(D, outliers, axis=0)
D = np.delete(D, outliers, axis=1)
# Let's look at the D-Matrix again, but this time with more focus on the ordinary instances:
KernelMethods.plotDTWDistancesMatrix(D)

# Next, we will cluster the lasers into intact and defect ones, and compute the silhouette coefficient of this clustering.
# This will lie in [-1, 1], the higher the better.
DataAnalysis.analyzeSilhouetteCoefficient(D, y)
# We observe a silhouette index of about 0.25, which is significantly higher than that of a random clustering, as shown.
# To visualize the cluster quality, we sort the D-matrix by laser quality labels.
KernelMethods.plotSortedDTWDistanceMatrix(D, y)
# As we see, the intact lasers lie closer to another than to the defect ones, and vice versa, whereas the defect lasers
# are not clustered as good. This gives some hope.




##################
### DTW-Kernel ###
##################

# We will use a kernel function to measure the similarity between the training instances. As a distance measure
# we take the DTW distance, so we can use D from before as a distance matrix.
dtwClassifier = KernelMethods.DTWClassifier(X, y, D, 1)
y_pred = dtwClassifier.predict(X)
print("Accuracy of DTW classifier:", sum(y == y_pred) / len(y))

# To find out what the best regularization parameter lambda is, we perform a nested cross validation.
dtwClassifierBestLbda, bestLbdaDTWKernel, avgRiskDTWKernel = KernelMethods.nestedCrossValidation(X, y, D, k=9)
print("Best lambda of best lambdas per test set:", bestLbdaDTWKernel, ", avg. risk overall:", avgRiskDTWKernel)
# We archieve a model with lbda = 3.65, an estimated risk of 0.

# Since our data is linearly separable (and will be in higher feature spaces), we now construct a Kernel Perceptron
kernelPerceptron = KernelMethods.KernelPerceptron(X, y, D)
print("Accuracy:", kernelPerceptron.accuracy(X, y))
print("entries != 0: ", np.where(kernelPerceptron.alpha != 0))
# As we expected, the perceptron is able to classify all training instances correctly. Although: the alpha vector
# looks suspiciously sparse, so we prefer the DTW model.




#############################
### LINEAR CLASSIFICATION ###
#############################

# To present a naive approach, we can construct a ridge classifier without regularization, using about 3/4 of the
# available training data. (Were we to use all training data, the model would classify all training data
# correctly, but that would be overfitting)
naiveLinearClassifier = LinearModels.RidgeClassifier(X[0:150], y[0:150], 0)
# This naive classifier classifies 96% of the test instances correctly:
print("Accuracy of naive linear classifier (lbda = 0):", naiveLinearClassifier.accuracy(X[150:200], y[150:200]))
# However this might still be a case of overfitting, so we investigate the weight vector theta:
print("L2 norm:", LinearModels.L2Regularizer(naiveLinearClassifier.thetaNormalized))
# It has an L2 norm of 0.49 (in the normalized form). As we will see, we can easily cut this norm in half,
# misclassifying only one instance of our given set, which seems like an acceptable tradeoff.

# With prior expiriments, we found out that the optimal value for lambda lies somewhere in the 40's / 50's.
# With this information, we perform a nested cross validation, with values for lambda between 35 and 55,
# in increments of 1, and k=10
bestRidgeClassifier, bestLbdaRidgeClassification, avgRiskDTWKernel = LinearModels.nestedCrossValidationRidgeClassification(X, y, k=9)
print("Nested cross validation yields the following model:")
print("Avg lambda:", bestLbdaRidgeClassification, ", avg risk:", avgRiskDTWKernel)
print("Accuracy on training set:", bestRidgeClassifier.accuracy(X, y), ", L2:", LinearModels.L2Regularizer(bestRidgeClassifier.getThetaNormalized()))
#
# Let's look at the precision/recall curve, regarding detecting defect lasers
precision, recall = Evaluation.precisionAndRecall(bestRidgeClassifier, X, y)
print("Precision:", precision, ", Recall:", recall)
Evaluation.precisionRecallCurve(bestRidgeClassifier, X, y)
# As we see, the "curve" is actually a rectangle - so precision and recall can each be forced to be 1, yielding
# a model that classifies all training instances correctly. The model that was outputted by the nested
# cross validation has a slightly lower precision, since we try to keep the L2 norm of theta low. So with this
# model, theta_0 is lower than the optimal model's theta_0, yielding a lower precision. But that is not too bad for
# this exercise, since the recall is more important, since we want to detect all defect lasers. And since the
# recall is 1, we're happy.

# Just for fun: since we know that we can construct a linear model with 100% accuracy, we know our data is
# linearly separable. Thus, we can construct a perceptron. Let's see how that goes:
perceptron = LinearModels.Perceptron(X[0:150], y[0:150])
print("Accuracy of Perceptron:", perceptron.accuracy(X[150:198], y[150:198]))
print("L2 norm of normalized theta:", LinearModels.L2Regularizer(perceptron.thetaNormalized))
# As we see, the accuracy of the perceptron is remarkably worse than that of the ridge classifier (0.86 vs. 0.98)
# This is probably because of the stochastic gradient descent, which is not as precise as an analytic solution.




#######################
### Descision Trees ###
#######################

from sklearn.ensemble import RandomForestClassifier
skForest = RandomForestClassifier()
skForest.fit(X[0:150], y[0:150])
y_pred = skForest.predict(X[150:198])
print("Accuracy of sklearn's random forest:", sum(y_pred==y[150:198])/len(y[150:198]))
Evaluation.plotFeatureImportances(skForest.feature_importances_)




###############
### SUMMARY ###
###############

minTrainingInstances = 2
maxTrainingInstances = 197
accsKernelTest = np.zeros(maxTrainingInstances - minTrainingInstances + 1)
accsRidgeTest = np.zeros(maxTrainingInstances - minTrainingInstances + 1)
accsForestTest = np.zeros(maxTrainingInstances - minTrainingInstances + 1)
nrsTrainingInstances = np.arange(minTrainingInstances, maxTrainingInstances+1)

for i in nrsTrainingInstances:
    training_indices = np.arange(0, i)
    test_indices = np.arange(i, 198)
    kernel_i = KernelMethods.DTWClassifier(X[training_indices],
                                           y[training_indices],
                                           D[np.ix_(training_indices, training_indices)],
                                           bestLbdaDTWKernel,
                                           X,
                                           D[:,training_indices])
    accsKernelTest[i - minTrainingInstances] = kernel_i.accuracy(X[test_indices], y[test_indices])
    ridge_i = LinearModels.RidgeClassifier(X[training_indices], y[training_indices], bestLbdaRidgeClassification)
    accsRidgeTest[i - minTrainingInstances] = ridge_i.accuracy(X[test_indices], y[test_indices])
    forest_i = RandomForestClassifier()
    forest_i.fit(X[training_indices], y[training_indices])
    y_pred_forest_i = forest_i.predict(X[test_indices])
    accsForestTest[i - minTrainingInstances] = sum(y_pred_forest_i == y[test_indices]) / len(y_pred_forest_i)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.plot(nrsTrainingInstances, accsKernelTest)
ax.plot(nrsTrainingInstances, accsRidgeTest)
ax.plot(nrsTrainingInstances, accsForestTest)
plt.xlabel("number of training instances")
plt.ylabel("Accuracy on remaining instances")
# ax.set_xscale("log")
plt.title("Comparison in accuracy between the models")
plt.legend(["Kernel Ridge Classification (DTW Kernel)", "Ridge Classification", "Random Forest (sklearn)"])
plt.show()