import numpy as np

# sigmoid function: 0 - 1 S型
def nonlin(x, deriv=False):
    if(deriv == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# input dataset
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
# print("Input data:")
# print(X, end="\n\n")

#output dataset
y = np.array([[0, 0, 1, 1]]).T
# print("Output data:")
# print(y, end="\n\n")

#seed random
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 1)) - 1
# print("Init whgiths:")
# print(syn0, end="\n\n")

for iter in range(1000):

    # forward propagetion
    layer_0 = X
    layer_1 = nonlin(np.dot(layer_0, syn0))
    # print("Layer1:")
    # print(layer_1)

    # how much did we miss?
    layer_1_error = y - layer_1
    # print("Error miss:")
    # print(layer_1_error)

    # multiply how much we missed by the slope of the sigmoid at the values in layer_1
    layer_1_delta = layer_1_error * nonlin(layer_1, True)
    # print("devirt of layer_1:")
    # print(nonlin(layer_1, True), end="\n\n")
    # print("layer_1_delta:")
    # print(layer_1_delta, end="\n\n")

    # update weights
    syn0 += np.dot(layer_0.T, layer_1_delta)
    # print("Update weights:")
    # print(syn0, end="\n\n")

print("Output After Training:")
print(layer_1)
print("Weights:")
print(syn0)




