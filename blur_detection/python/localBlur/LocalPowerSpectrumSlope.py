import cv2
import numpy as np
from likematlab import im2col, GenerateF, calculateSf, GenerateSf, rearrange

def LocalPowerSpectrumSlope(img,patchsize):
    im_height, im_width = img.shape
    offset = (patchsize - 1)/2

    [If, IC] = GenerateF(offset,offset)
    IC_log = np.log(IC)
    IC_log_new = np.unique(np.round(IC_log / 0.2)*0.2)

    im_col = im2col(img, [patchsize,patchsize])
    im_col = im_col / np.sum(im_col,1,keepdims=True)
    q = np.zeros((1,im_col.shape[1]))

    for i in range(im_col.shape[1]):
        lsf = GenerateSf(im_col[:,i].reshape(patchsize,patchsize), If, IC, offset, offset)
        lsf_new = rearrange(IC_log, IC_log_new, lsf)
        lsf_new = np.log(lsf_new[0:-1])
        idx = ~(np.isnan(lsf_new) + np.isinf(lsf_new))
        alf_local = np.sum(lsf_new[idx])
        q[0,i] = alf_local

    q = q.reshape(im_width-patchsize+1,im_height-patchsize+1)
    q = np.pad(q,offset,'edge')
    q = np.transpose(q)

    return q