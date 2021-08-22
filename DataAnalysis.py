from matplotlib import pyplot as plt
import numpy as np

def plotIndividualIntensitySeries(X, y, indices = []):
    if indices == []:
        indices = np.arange(X.shape[0])
    plt.subplot(1, 2, 1)
    for i in indices:
        if y[i] == -1:
            plt.plot(np.arange(60), X[i,:], alpha=0.35)
    plt.ylabel("intensity")
    plt.xlabel("t in s")
    plt.title("Intact lasers")
    plt.subplot(1, 2, 2)
    for i in indices:
        if y[i] == 1:
            plt.plot(np.arange(60), X[i,:], alpha=0.35)
    plt.ylabel("intensity")
    plt.xlabel("t in s")
    plt.title("Faulty lasers")
    plt.tight_layout()
    plt.show()

def silhouetteCoefficient(D, C):
    s = np.zeros(D.shape[0])
    for i in range(D.shape[0]):
        a = D[i,C == C[i]].mean()
        b = D[i,C != C[i]].mean()
        s[i] = (b-a)/max(a, b)
    return s.mean()

def analyzeSilhouetteCoefficient(D, C, tries=1000):
    silhouette = silhouetteCoefficient(D, C)
    print("Silhouette of best cluster:", silhouette)
    silhouettes = np.zeros(tries)
    for i in range(tries):
        clusterCandidate = np.random.randint(2, size=len(C))
        silhouettes[i] = silhouetteCoefficient(D, clusterCandidate)
    print("Mean of random clusters:", silhouettes.mean())
    print("Min  of random clusters:", silhouettes.min())
    print("Max  of random clusters:", silhouettes.max())