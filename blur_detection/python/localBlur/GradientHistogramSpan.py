import cv2
import numpy as np
from likematlab import EM_GMM, im2col

def GradientHistogramSpan(img, patchsize):
    im_height, im_width = img.shape
    offset = int((patchsize -1)/2)

    # Compute image gradient
    dx = np.c_[np.diff(img),-img[:,-1]]
    dy = np.r_['0,2',np.diff(img,axis=0),-img[-1,:]]

    mag = np.sqrt(np.square(dx) + np.square(dy))
    mag_col = im2col(mag,[patchsize,patchsize])
    mag_col_dup = np.hstack((mag_col,-mag_col))

    num = mag_col.shape[1]
    sigma1 = np.zeros((1,num))

    for i in range(num):
        # time consuming 부분
        [V1,V2] = EM_GMM(mag_col_dup[:,i])
        s0 = np.sqrt(V1)
        s1 = np.sqrt(V2)
        if s0 > s1 :
            temp = s0; s0 = s1; s1 = temp

        sigma1[0,i] = s1

    q = np.reshape(sigma1, [im_height-patchsize+1,im_width-patchsize+1])
    q = q[4:(im_height-patchsize+1-3),4:(im_width-patchsize+1-3)]
    q = np.pad(q,offset+3,'edge')

    return q

