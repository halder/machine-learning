import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class NaiveBayes:
    def fit(self, X, y, smoothing=10e-3): # fit initializes class
        self.gaussians = dict() # holds mean and var for given line -> used to calculate p(x|C) = N(mean, var)
        self.priors = dict() # probability of event C -> p(C) with C = "spam"

        labels = set(y) # set of target labels; 0 to 9

        # columns in X equals the target vector; e.g. column 0 equals target "0" etc.
        for l in labels:
            current_x = X[y == l] # select current_x from X where the column equals the current target label l

            self.gaussians[l] = {
                "mean":current_x.mean(axis=0), # calc. mean for current_x
                "var":current_x.var(axis=0) + smoothing # calc. var (+smoothing) for current_x
            }
            self.priors[l] = len(y[y == l]) / len(y) # priors for the given label l; discrete (divide distinct label by number of labels)

    def predict(self, X):
        N, D = X.shape # get length and width of X
        print("len N: {}; dimension/width D: {}".format(N, D))
        K = len(self.gaussians) # equals amount of targets // width of X
        print("len K: {}".format(K))
        P = np.zeros((N, K)) # target matrix of size N (length) by K (width)

        for l, g in self.gaussians.items():
            mean, var = g["mean"], g["var"]

            # calculate posterior probability for each row:

            # mvn.logpdf calculates p(X|C) = N(x1; mean1, var1)*N(x2; m2, v2)*...*N(xn; mn, vn)
            # with x being the different possible target values
            # taking log reduces computational resources required; log monotonically increasing -> same result
            P[:, l] = mvn.logpdf(X, mean=mean, cov=var) + np.log(self.priors[l]) # added to P in column of corresponding label l

            print("P", P[:5])
            print("argmax(P)", np.argmax(P[:5], axis=1))
        return np.argmax(P, axis=1) # return vector of l's with the highest argmax of all columns in P

    def score(self, X, y):
        P = self.predict(X)
        return np.mean(P == y), P
