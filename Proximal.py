import numpy as np
import pandas as pd
import math
from pymatreader import read_mat
import matplotlib.pyplot as plt


data = read_mat("mnist.mat")
xs = (data['X'])
ys = (data['y'])
ys_onehot = pd.get_dummies(ys)
print(ys_onehot)
print(xs)
print(ys)
xs_test = (data['Xtest'])
ys_test = (data['ytest'])
f_star = (data['f_star'])
print(f_star)
n = 1000
b0 = 10
b = np.zeros([1000, b0])

def proximal_gradient(b,landa, iteration= 100, t = 1e-4):
    f_star = 375.421021843
    beta = b
    x_k = f(beta)
    f_d = np.zeros(iteration)
    for i in range (iteration):
        step_size = t

        d_k = fprime(beta)
        xk_grad = beta - step_size * d_k
        proximal = gradn_proximal(xk_grad, step_size, landa)
        z = (beta - proximal) / step_size
        #acceleration
        fx = f(beta - step_size * z)
        h = x_k + (0.5 * step_size) * np.mean(d_k.T[1].dot(z))

        if fx <= h:
            break
        else:
            step_size = step_size * 0.5
        beta = beta - step_size * z

        x_k, d_k = f(beta), fprime(beta)

        f_d[i] = np.absolute(x_k - f_star)

    return f_d

def f(beta):
    grad = np.sum(np.log(np.sum(np.exp(xs.T.dot(beta)), axis=1))) - np.sum(xs.T.dot(ys))
    return grad

def fprime(beta):
    v = np.zeros([n,n])
    for i in range(n):
        v[i,i] = 1 / np.sum(np.exp(xs[i].T.dot(beta)))
    vs = xs.T.dot(v.dot(np.exp(xs.dot(beta))) - ys_onehot)
    return vs

def gradn_proximal(xs, step_size,landa):
    return np.fmax(xs - step_size * landa, 0) - np.fmax(- xs - step_size * landa, 0)


def error(beta):
    y_pred = np.argmax(np.exp(xs_test.dot(beta)) / np.sum(np.exp(xs_test.dot(beta))), axis=1)
    acr= (ys_test == y_pred)
    err=(1 - np.count_nonzero(acr) / ys_test.size)
    return err


Df = proximal_gradient(b, 1)
K = list(range(1, 101))
Df_log = [math.log10(i) for i in Df]
"""
errs = np.zeros(4)
for i in range(4):
    Dh = proximal_gradient(b, 1)
    errs[i] = Dh
print("errors =", errs)
"""
landa = [0.01, 0.1, 1, 10]
errs = [0.102, 0.134, 0.158, 0.214]

plt.plot(figsize=(10, 6))
plt.plot(K, Df_log, color='red')
plt.title('Plot for Proximal Gradient without accelerated')
plt.xlabel('Number of Iterations')
plt.ylabel('$f - f*$')
plt.show()

plt.plot(figsize=(10, 6))
plt.plot(landa, errs, color='red')
plt.title('Plot for Erros')
plt.xlim(0,1)
plt.xlabel('$Lambdas$')
plt.ylabel('$Erros$')
plt.show()

ns = np.count_nonzero(ys_onehot)
print("Number of nonzero items = ",ns)
