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
    print("Dim 1")
    plt.figure(1)

    #calcul de la solution
    w0 = np.array([20])
    lr = 0.7
    res_1 = gd2_momentum(w0, grad, lr, max_iter=33)

    #plot
    x = np.linspace(-20, 20, 100)
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i] = f(np.array([x[i]]))
    res_1_y = np.zeros(res_1.shape)
    for i in range(res_1.shape[0]):
        res_1_y[i] = f(np.array([res_1[i]]))

    plt.plot(x,y)
    plt.plot(res_1, res_1_y, 'o-', c='red')
    plt.title("Dimension 1")

    print("w final: {}".format(res_1[-1]))
    print("")

    ## Dim 2
    plt.figure(2)
    w0 = np.array([-20, 20])
    lr = 0.6
    res_2 = gd2_momentum(w0, grad, lr, max_iter=22)

    #plot
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)
    z = np.zeros((x.shape[0], y.shape[0]))
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = f(np.array([x[i], y[j]]))
    print(np.min(z))
    print(np.max(z))
    levels = [790, 800, 810, 815, 820, 825, 830, 835, 840, 845, 850, 860, 870, 880, 890]
    c = plt.contour(X, Y, z, levels)
    plt.plot(res_2[:, 0], res_2[:, 1], 'o-', c='red')
    plt.title("Dimension 2")

    print("w final: {}".format(res_2[-1, :]))
    # print(res_2)
    print("")

    ## Dim 10
    w0 = np.array([20, 20, -20, 20, -20, 20, 20, 20, 20, 20])
    lr = 0.1
    res_10 = gd2_momentum(w0, grad, lr, max_iter=10)
    print("w final: {}".format(res_10[-1, :]))
    plt.show()

if __name__ == "__main__":
    main()
