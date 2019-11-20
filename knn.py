import pandas as pd
import numpy as np
from sortedcontainers import SortedList

class KNN:
    def __init__(self, k):
        self.k = k # initialize k

    def fit(self, X, y): # train points
        print("fitting {}-NN...".format(self.k))
        self.X = X # initialize values X
        self.y = y # initialize target Y

    def predict(self, X):
        print("predicting {}-NN...".format(self.k))
        y = np.zeros(len(X)) # empty prediction vector

        # iterating over self.X over X == comparing each datapoint to each other
        for i, x in enumerate(X): # iterate over test points
            k_nearest = SortedList() # init empty sorted list, take predicted target from there. will hold k values.
            for j, xt in enumerate(self.X): # iterate over trained points
                diff = x-xt # calculate distance X to each Xt in self.X
                distance = diff.dot(diff) # calculate dot product (essentially distance d; squared difference)

                if len(k_nearest) < self.k: # first k distances get added to sortedlist without condition
                    k_nearest.add((distance, self.y[j])) # distance + target at pos j
                else:
                    # if sortedlist holds k items, compare current distance to the last entry;
                    # since it's a sortedlist the item at position [-1](last) will be of the highest value (longest distance)
                    if distance < k_nearest[-1][0]:
                        del k_nearest[-1] # if current distance is smaller than [-1], replace it
                        k_nearest.add((distance, self.y[j]))

                votes = dict() # initiate empty dictionary that will hold votes
                for _, characteristic in k_nearest: # iterate over sortedlist; only the characteristic (target) is of importance
                    votes[characteristic] = votes.get(characteristic, 0) + 1 # key = characterstic, value = count within sortedlist


                max_votes = 0 # counter for target characteristic
                dominant_char = -1 # placeholder target

                for char, n_votes in votes.items():
                    if n_votes > max_votes:
                        max_votes = n_votes
                        dominant_char = char # set target for x to the char. with max votes in "votes"

                y[i] = dominant_char # add target(x) to target vector y at pos i
        return y

    def score(self, X, Y):
        print("calculating score of {}-NN...".format(self.k))
        prediction = self.predict(X)
        return np.mean(prediction == Y)
