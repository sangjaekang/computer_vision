import cv2
import numpy as np

def im2col(A, BSZ):
    # Parameters
    A = np.transpose(A)
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::1]

def EM_GMM(X):
    # Compute expectation for GMM with two Gaussian Component

    # Initialize parameters
    V1 = 0.5
    V2 = 0.0001
    pi = 0.5
    N  = np.size(X)

    pi_init = 2 *pi
    maxiter = 100
    count    = 0

    XxX = np.square(X)

    while (abs(pi_init-pi)/pi_init > 1e-3 and count < maxiter):
        count  += 1
        pi_init = pi

        # Expectatino Step
        pi_x_GPV2 = pi * GaussianProb(XxX,V2)
        gamma = pi_x_GPV2 / ((1-pi)*GaussianProb(XxX,V1) + pi_x_GPV2)
        # Maximization Step

        V1 = np.matmul(np.transpose(1-gamma),XxX) / np.sum(1-gamma)
        V2 = np.matmul(np.transpose(gamma),XxX)   / np.sum(gamma)
        pi = np.sum(gamma)/N

    return [V1,V2]

def GaussianProb(XxX, V):
    return np.exp(-(XxX)/(2*V)) / np.sqrt(2*np.pi*V)
