import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gradDescent(t0, t1, alpha, iterations, n):
    for i in range(iterations):
        tempo0 = (1 / n) * np.sum([t0 + t1 * x[j] - y[j] for j in range(n)])
        tempo1 = (1 / n) * np.sum([(t0 + t1 * x[j] - y[j]) * x[j] for j in range(n)])

        temp0 = t0 - (alpha * tempo0)
        temp1 = t1 - (alpha * tempo1)

        t0 = temp0
        t1 = temp1

    return t0, t1

def stocDescent(t0, t1, alpha, iterations, n):
    for i in range(iterations):
        for j in range(n):
            tempo0 = t0 + t1 * x[j] - y[j]
            tempo1 = (t0 + t1 * x[j] - y[j]) * x[j]

            temp0 = t0 - (alpha * tempo0)
            temp1 = t1 - (alpha * tempo1)

            t0 = temp0
            t1 = temp1

    return t0, t1

def yPredDescent(x_test):
    newx = x_test
    newy = gradDescent(t0, t1, alpha, iterations, n)[0] + gradDescent(t0, t1, alpha, iterations, n)[1] * newx
    return newy

def yPredStocDescent(x_test):
    newx = x_test
    newy = stocDescent(t0, t1, alpha, iterations, n)[0] + stocDescent(t0, t1, alpha, iterations, n)[1] * newx
    return newy

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
x = x.flatten()
y = y.flatten()

iterations = 10000
alpha = 0.01
t0 = 0
t1 = 0
n = len(x)

x_test = np.array([0, 1.1, 2.6, 4.9, 9.7, 10.5])

print(gradDescent(t0, t1, alpha, iterations, n)[0], gradDescent(t0, t1, alpha, iterations, n)[1])
print(yPredDescent(x_test))

print(stocDescent(t0, t1, alpha, iterations, n)[0], stocDescent(t0, t1, alpha, iterations, n)[1])
print(yPredDescent(x_test))


plt.plot(x, y, 'ro', label="Data")
plt.plot(x_test, yPredDescent(x_test), 'b-', label="Batch Descent")
plt.plot(x_test, yPredStocDescent(x_test), 'g-', label="Stochastic Descent")
plt.legend(loc="upper left")
plt.show()




