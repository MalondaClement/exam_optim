import numpy as np
import matplotlib.pyplot as plt

def f(w):
    res = 418.9829 * w.shape[0]
    sum = 0
    for i in range(w.shape[0]):
        sum = sum + w[i] * np.sin(np.sqrt(np.abs(w[i])))
    return res - sum

def grad(w):
    res = 0
    for i in range(w.shape[0]):
        res = res + (w[i]**2 + np.cos(np.sqrt(np.abs(w[i])))) / (2*np.abs(w[i])**(3/2)) + np.sin(np.sqrt(np.abs(w[i])))
    return res

def gd2_momentum(x, grad, alpha, beta=0.9, max_iter=10):
    xs = np.zeros((1 + max_iter, x.shape[0]))
    xs[0, :] = x
    v = 0
    for i in range(max_iter):
        v = beta*v + (1-beta)*grad(x)
        vc = v/(1+beta**(i+1))
        x = x - alpha * vc
        xs[i+1, :] = x
    return xs

def main():
    ## Dim 1

    ## Dim 1

    ## Dim 2

if __name__ == "__main__":
    main()
