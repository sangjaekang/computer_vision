import cv2
import numpy as np
from likematlab import im2col
from scipy import stats

def LocalKurtosis(img,patchsize):
    img_height,img_width = img.shape
    offset = (patchsize-1)/2

    x_start = offset+1; x_end = im_width-offset;
    y_start = offset+1; y_end = im_height-offset;
    datasize = (x_end-x_start+1)*(y_end-y_start+1)

    # calculate image gradient
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

    sobelx_col = im2col(sobelx,[patchsize,patchsize])
    sums_dx_col = np.sum(sobelx_col,0,keepdims=True)
    sums_dx_col[sums_dx_col == 0] = np.nan
    sobelx_col = sobelx_col / sums_dx_col

    sobely_col = im2col(sobely,[patchsize,patchsize])
    sums_dy_col = np.sum(sobely_col,0,keepdims=True)
    sums_dy_col[sums_dy_col == 0] = np.nan
    sobely_col = sobely_col / sums_dy_col

    # calculate kurtosis
    dx_kurtosis = stats.kurtosis(sobelx_col,axis=0,fisher=False)
    dy_kurtosis = stats.kurtosis(sobely_col,axis=0,fisher=False)

    dx_kurtosis = np.reshape(dx_kurtosis, [im_width-patchsize+1, im_height-patchsize+1])
    dy_kurtosis = np.reshape(dy_kurtosis, [im_width-patchsize+1, im_height-patchsize+1])

    dx_kurtosis = np.log(np.pad(dx_kurtosis,offset,mode='edge'))
    dy_kurtosis = np.log(np.pad(dy_kurtosis,offset,mode='edge'))

    result = np.transpose(np.minimum(dx_kurtosis,dy_kurtosis))

    return result