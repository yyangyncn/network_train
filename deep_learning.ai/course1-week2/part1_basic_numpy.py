import math

# 1.1
# basic_sigmod
def basic_sigmod(x):
    s = 1 / (1 + math.exp(-x))
    return s

print(basic_sigmod(3))

import numpy as np

def sigmod(x):
    s = 1 / (1 + np.exp(-x))
    return s
x = np.array([1, 2, 3])
print(sigmod(x))

# 1.2 sigmod gradient
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds
print("derivative:", sigmoid_derivative(x))

# 1.3 Reshaping Array
def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
print ("image2vector(image) = " + str(image2vector(image)))

# 1.4 Normalizing rows
def nomalize_row(x):
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    # devide x by its norm
    x = x / x_norm
    return x
x = np.array([[0, 3, 4], [1, 6, 4]])
print("nomalize_row(x) = " + str(nomalize_row(x)))

# 1.5 broadcasting and softmax
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s
print("softmax(x) = " + str(softmax(x)))

# 2) L1 and L2 loss function
# L1(ŷ ,y)=∑i=0,m |y(i)−ŷ (i)|
def L1(yhat, y):
    loss = np.sum(np.abs(y - yhat))
    return loss
yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

def L2(yhat, y):
    loss = np.sum(np.dot(y-yhat, y-yhat))
    return loss
print("L2 = " + str(L2(yhat,y)))