import numpy as np
import sys

# Generate random examples
# ----------------------------------------------------------
inputs = 9
outputs = 5
data = 100

m = data
X = (np.random.random((inputs, data)) - 0.5) * 2
ys = np.random.randint(0, outputs, size=data)
Y = np.zeros((outputs, data))
for i in range(data):
    Y[ys[i]][i] = 1
# ----------------------------------------------------------

# hyperparameters
epoch = 1000
alpha = 0.01
decay_rate = 1
beta_1 = 0.9
beta_2 = 0.999
lambd = 0.5
batch = 64 # batch size, power of 2
layers = [None, 5, 5, 5, 5, outputs]
epsilon = 1e-8

# activation functions
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0, keepdims=True)
def d_softmax(z):
    summ = np.sum(np.exp(z), axis=0, keepdims=True)
    return (np.exp(z) * summ - np.exp(2*z)) / (summ ** 2)

def ReLU(z):
    return np.maximum(0, z)
def d_ReLU(z):
    return np.where(z < 0, 0.0, 1.0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))
def d_sigmoid(z):
    return sigmoid(z)*(1 - sigmoid(z))

def tanh(z):
    return np.tanh(z)
def d_tanh(z):
    return 1 - tanh(z)**2

# cost functions
def cost(Y_hat, Y):
    return -np.sum(Y * np.log(Y_hat + epsilon)) / Y.shape[1]
def d_cost(Y_hat, Y):
    return -(Y / (Y_hat + epsilon))

# subtract mean
mu = np.sum(X) / m
X = X - mu

# Normalize Variance
sigma = np.sum(X**2)/m
X /= sigma

# minibatch
layers[0] = inputs
layer = len(layers) - 1
batch_num = int(m / batch + 1)
X = np.array_split(X, batch_num, 1)
Y = np.array_split(Y, batch_num, 1)

# cache
A = [X] + [None] * layer
Z = [None] + [None] * layer
g = [None] + [ReLU] * (layer - 1) + [sigmoid]
dg = [None] + [d_ReLU] * (layer - 1) + [d_sigmoid]
W = [None]
VdW = [None]
SdW = [None]
Beta = [None]
VdBeta = [None]
SdBeta = [None]
Gama = [None]
VdGama = [None]
SdGama = [None]
rate = alpha
epoch_num = 0

# Weight Initialization
for l in range(1, layer + 1):
    # weight = weight * Var(W)
    # ReLU: Var(W) = sqrt(2 / n[l-1])
    # tanh: Var(W) = sqrt(1 / n[l-1])
    #             or sqrt(2 / (n[l-1] + n[l]))
    W.append(np.random.random((layers[l], layers[l-1]))*np.sqrt(2/layers[l-1]))
    VdW.append(np.zeros((layers[l], layers[l-1])))
    SdW.append(np.zeros((layers[l], layers[l-1])))

    Beta.append(np.zeros((layers[l], 1)))
    VdBeta.append(np.zeros((layers[l], 1)))
    SdBeta.append(np.zeros((layers[l], 1)))

    Gama.append(np.zeros((layers[l], 1)))
    VdGama.append(np.zeros((layers[l], 1)))
    SdGama.append(np.zeros((layers[l], 1)))

# learning
for i in range(1, epoch + 1):
    L = 0
    # learning rate decay
    rate = 1 / (1 + decay_rate * epoch_num) * alpha
    for t in range(batch_num):
        A[0] = X[t]
        for l in range(1, layer + 1):
            Z[l] = np.dot(W[l], A[l - 1])
            mu = np.sum(Z[l]) / m
            sigma = np.sum((Z[l] - mu)**2) / m
            Z_norm = (Z[l] - mu) / (sigma + epsilon) ** (0.5)
            Z_tilde = Gama[l] * Z_norm + Beta[l]
            A[l] = g[l](Z_tilde)
        Y_hat = A[layer]
        L += cost(Y_hat, Y[t])

        dA = d_cost(Y_hat, Y[t])
        for l in range(layer, 0, -1):
            # recalculate constants
            mu = np.sum(Z[l]) / m
            sigma = np.sum((Z[l] - mu)**2) / m
            Z_norm = (Z[l] - mu) / (sigma + epsilon) ** (0.5)
            Z_tilde = Gama[l] * Z_norm + Beta[l]
            
            # Calculating derivative
            dZ_tilde = dA * dg[l](Z_tilde)
            dZ_norm = dZ_tilde * Gama[l]
            dZ = dZ_norm * (sigma + epsilon) ** (-0.5)
            dW = np.dot(dZ, A[l-1].T) / m
            dGama = np.sum(dZ_tilde * Z_norm, axis=1, keepdims=True) / m
            dBeta = np.sum(dZ_tilde, axis=1, keepdims=True) / m
            
            # Regularized derivative
            dW += lambd / m * W[l]
            dGama += lambd / m * Gama[l]
            
            dA = np.dot(W[l].T, dZ)
            
            # Momentum & RMSprop
            VdW[l] = beta_1 * VdW[l] + (1-beta_1)*dW
            VdBeta[l] = beta_1 * VdBeta[l] + (1-beta_1)*dBeta
            VdGama[l] = beta_1 * VdGama[l] + (1-beta_1)*dGama
            SdW[l] = beta_2 * SdW[l] + (1-beta_2)*dW**2
            SdBeta[l] = beta_2 * SdBeta[l] + (1-beta_2)*dBeta**2
            SdGama[l] = beta_2 * SdGama[l] + (1-beta_2)*dGama**2
            
            # Required: Bias correction
            VdW_corrected = VdW[l] / (1 - beta_1**(epoch_num*batch_num+t+1))
            VdBeta_corrected = VdBeta[l] / (1 - beta_1**(epoch_num*batch_num+t+1))
            VdGama_corrected = VdGama[l] / (1 - beta_1**(epoch_num*batch_num+t+1))
            SdW_corrected = SdW[l] / (1 - beta_2**(epoch_num*batch_num+t+1))
            SdBeta_corrected = SdBeta[l] / (1 - beta_2**(epoch_num*batch_num+t+1))
            SdGama_corrected = SdGama[l] / (1 - beta_2**(epoch_num*batch_num+t+1))
            
            # Gradient descent
            W[l] = W[l] - VdW_corrected / (SdW_corrected**(1/2) + epsilon) * rate
            Gama[l] = Gama[l] - VdGama_corrected / (SdGama_corrected**(1/2) + epsilon) * rate
            Beta[l] = Beta[l] - VdBeta_corrected / (SdBeta_corrected**(1/2) + epsilon) * rate
    
    epoch_num += 1
    
    # Regularized cost
    J = L / m
    for l in range(1, layer + 1):
        J += lambd / (2*m) * np.sum(W[l]**2)
    
    if((int(i / epoch * 100)) % 20 == 0 and int(i * 100 / epoch) == i * 100 / epoch):
        sys.stdout.write(str(int(i / epoch * 100)) + "%\nloss: " + str(J) + "\n\n")
