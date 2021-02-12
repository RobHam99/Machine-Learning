import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


def logRegStochastic(b0, b1, b2, X1, X2, Y, iterations, alpha):
    for i in range(iterations):
        for j in range(len(X1)):
            prediction = 1 / (1 + np.exp(-(b0 + b1*X1[j] + b2*X2[j])))
            b0 = b0 + alpha * (Y[j] - prediction) * prediction * (1 - prediction) * 1
            b1 = b1 + alpha * (Y[j] - prediction) * prediction * (1 - prediction) * X1[j]
            b2 = b2 + alpha * (Y[j] - prediction) * prediction * (1 - prediction) * X2[j]

    return b0, b1, b2

def decidePred(test_X1, test_X2):
    barr = logRegStochastic(b0, b1, b2, X1, X2, Y, iterations, alpha)
    output = np.empty(len(test_X1))
    prob = np.empty(len(test_X1))
    decide = np.empty(len(test_X1))
    n = len(test_X1)
    for i in range(n):
        output[i] = barr[0] + barr[1] * test_X1[i] + barr[2] * test_X2[i]
        prob[i] =  1 / (1 + np.exp(-output[i]))
    for i in range(n):
        if prob[i] < 0.5:
            decide[i] = 0
        elif prob[i] >= 0.5:
            decide[i] = 1

    return decide

sc = StandardScaler()

data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:300, :-1]
Y = data.iloc[:300, -1]
test_X = data.iloc[300:, :-1]
test_Y = data.iloc[300:, -1].to_numpy().flatten()

X = sc.fit_transform(X)
test_X = sc.transform(test_X)

X1 = X[:, :-1]
X2 = X[:, -1]
test_X1 = test_X[:, :-1]
test_X2 = test_X[:, -1]

b0 = 0
b1 = 0
b2 = 0
alpha = 0.003
iterations = 1000

pred_Y = decidePred(test_X1, test_X2)
cm = confusion_matrix(test_Y, pred_Y )

print(logRegStochastic(b0, b1, b2, X1, X2, Y, iterations, alpha))
print(np.concatenate((pred_Y.reshape(len(test_Y), 1), test_Y.reshape(len(test_Y), 1)), 1))
print(cm)
print(accuracy_score(test_Y, pred_Y))

plt.plot(X1, X2, 'ro', test_X1, test_X2, 'bo')
plt.show()