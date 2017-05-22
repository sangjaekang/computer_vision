import cv2
import numpy as np
from likematlab import im2col

def LocalKurtosis(img,patchsize):
    img_height,img_width = img.shape
    offset = (patchsize-1)/2

    x_start = offset+1; x_end = im_width-offset;
    y_start = offset+1; y_end = im_height-offset;
    datasize = (x_end-x_start+1)*(y_end-y_start+1)

    # Compute image gradient
    dx = np.c_[np.diff(img),-img[:,-1]]
    dy = np.r_['0,2',np.diff(img,axis=0),-img[-1,:]]

    # Rearrange image to patches
    dx_col = im2col(dx,[patchsize,patchsize])
    dx_col = dx_col / np.sum(dx_col,1,keepdims=True)
    dy_col = im2col(dy,[patchsize,patchsize])
    dy_col = dy_col / np.sum(dy_col,1,keepdims=True)

    # Compute Kurtosis
    normXsquare = np.square(dx_col - np.mean(dx_col))
    normYsquare = np.square(dy_col - np.mean(dy_col))

    qx = np.mean(np.square(normXsquare),axis=0,keepdims=True) / np.square(np.mean(normXsquare,axis=0,keepdims=True))
    qy = np.mean(np.square(normYsquare),axis=0,keepdims=True) / np.square(np.mean(normYsquare,axis=0,keepdims=True))

    qx = np.reshape(qx, [im_height-patchsize+1, im_width-patchsize+1])
    qy = np.reshape(qy, [im_height-patchsize+1, im_width-patchsize+1])

    # Normalize for output
    qx = np.log(np.pad(qx,offset,mode='edge'))
    qy = np.log(np.pad(qy,offset,mode='edge'))

    qx[np.isnan(qx)] = np.min(np.min(qx[~np.isnan(qx)]))
    qy[np.isnan(qy)] = np.min(np.min(qy[~np.isnan(qy)]))

    return np.minimum(qx,qy)