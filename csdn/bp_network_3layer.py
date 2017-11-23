import numpy as np

def nonlin(x, deriv=False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(1 - x))

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],[1],[1],[0]])
print(np.array([0, 0, 1, 1]).T)
print(np.array([[0, 0, 1, 1]]).T)

np.random.seed(1)

# randomly initialize
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):

    # Feed forward
    layer_0 = X
    layer_1 = nonlin(np.dot(layer_0, syn0))
    layer_2 = nonlin(np.dot(layer_1, syn1))

    layer_2_error = y - layer_2
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(layer_2_error))))
    layer_2_delta = layer_2_error * nonlin(layer_2, deriv=True)

    layer_1_error = layer_2_delta.dot(syn1.T)
    layer_1_delta = layer_1_error * nonlin(layer_1, deriv=True)

    syn1 += layer_1.T.dot(layer_2_delta)
    syn0 += layer_0.T.dot(layer_1_delta)