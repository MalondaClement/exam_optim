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
    lr = 0.7

    ## Dim 1
    print("Dim 1")
    plt.figure(1)

    #calcul de la solution
    w0 = np.array([20])
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

    print("w final: {}".format(res_1[-1]))
    print("")

    ## Dim 2
    plt.figure(2)
    w0 = np.array([-20, 20])
    res_2 = gd2_momentum(w0, grad, lr, max_iter=33)

    #plot
    # x = np.linspace(-20, 20, 100)
    # y = np.linspace(-20, 20, 100)
    # X, Y = np.meshgrid(x, y)
    # z = np.zeros((x.shape[0], y.shape[0]))
    # for i in range(x.shape[0]):
    #     for j in range(y.shape[0]):
    #         z[i, j] = f(np.array([x[i], y[j]]))
    # levels = [0.1,1,2,4,9, 16, 25, 36, 49, 64, 81, 100, 1000]
    # c = plt.contour(X, Y, z, levels)

    print("")

    ## Dim 10
    plt.figure(3)
    w0 = np.array([20, 20, -20, 20, -20, 20, 20, 20, 20, 20])
    # res_10 = gd2_momentum(w0, grad, lr, max_iter=33)

    plt.show()

if __name__ == "__main__":
    main()
