import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def xMean(x, n):
    x_mean = np.sum(x) / n
    return x_mean

def yMean(y, n):
    y_mean = np.sum(y) / n
    return y_mean

def grad(x, y, n):
    m = sum((x - xMean(x, n)) * (y - yMean(y, n))) / sum((x - xMean(x, n)) ** 2)
    return m


def yInt(x, y, n):
    b = yMean(y, n) - grad(x, y, n) * xMean(x, n)
    return b

def yPred(x_test):
    newx = x_test
    newy = grad(x, y, n) * newx + yInt(x, y, n)
    return newy

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x = x.flatten()
y = y.flatten()

n = len(x)

x_test = np.array([0, 1.1, 2.6, 4.9, 9.7, 12, 15])

print(yInt(x, y, n), grad(x, y, n))
print(yPred(x_test))
plt.plot(x, y, 'ro', x_test, yPred(x_test), 'b-')
plt.show()