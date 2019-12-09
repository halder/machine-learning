import numpy as np

v = [1, 1]

def dj(w):
    return [2 * w[0], 4 * w[1] ** 3]

def gd(w, iterations, l):
    for i in range(iterations):
        w[0] = w[0] - l*dj(w)[0]
        w[1] = w[1] - l*dj(w)[1]
        print(w[0], w[1])

gd(v,100,0.49)
