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

def GenerateF(height,width):
    kheight = range(1,int(height)+1)
    kwidth  = range(1,int(width)+1)
    [u,v] = np.meshgrid(kwidth,kheight)
    f = np.hypot(u, v)
    C = np.unique(f)
    f = np.round(f)
    return (f,C)

def calculateSf(s, C, f):
    sf = np.zeros(C.shape)
    for i in range(np.size(C)):
        sf[i] = np.sum(s[f==C[i]])
    return sf

def GenerateSf(im, f, C, half_height, half_width):
    height, width = im.shape
    s = np.abs(np.fft.fft2(im))
    s = np.square(s[0:half_height,0:half_width]) / (half_height * half_width)
    sf = calculateSf(s, C, f)
    return sf

def rearrange(C, C_new, f):
    data = np.zeros(C_new.shape)
    idx = 0
    for i in range(len(f)):
        if (C[i] >= C_new[idx+1]):
            idx = idx+1
        data[idx] = data[idx] + f[i]
    return data