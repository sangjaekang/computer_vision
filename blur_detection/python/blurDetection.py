import numpy as np
import pandas as pd
import cv2
import os
from localBlur import LocalKurtosis, GradientHistogramSpan, LocalPowerSpectrumSlope
import sys
from multiprocessing import Queue

# Naive-Bayes classifier value
params = [[(1.03836469274083,0.360097973190199),(0.947234770222116,0.441889606327453)],
[(0.089113883910261,0.0796449095925369),(0.0259784177855934,0.0357503808612129)],
[(-26.358860419625,1.98849960841472),(-28.3796098948851,3.004809639)]]

prior = [0.3,0.7]

def main(raw_image, gt_image, output_path):
    img = cv2.imread(raw_image, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img.astype('double'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    patchsize = 11

    img_gt = cv2.imread(gt_image, cv2.IMREAD_GRAYSCALE)
    img_gt[img_gt==255] = 1

    im_height, im_width = img.shape
    offset = int((patchsize -1)/2)
    x_start = offset+1; x_end = im_width-offset;
    y_start = offset+1; y_end = im_height-offset;
    datasize = (x_end-x_start+1)*(y_end-y_start+1)

    print("{}-Localkurtosis doing...".format(raw_image))
    q1 = LocalKurtosis.LocalKurtosis(img, patchsize)
    print("{}-GradientHistogramSpan doing...".format(raw_image))
    q2 = GradientHistogramSpan.GradientHistogramSpan(img, patchsize)
    print("{}-LocalPowerSpectrumSlope doing...".format(raw_image))
    q3 = LocalPowerSpectrumSlope.LocalPowerSpectrumSlope(img, patchsize)

    data = np.zeros((datasize,3))
    data[:,0] = np.ravel(q1[y_start:y_end+1,x_start:x_end+1])
    data[:,1] = np.ravel(q2[y_start:y_end+1,x_start:x_end+1])
    data[:,2] = np.ravel(q3[y_start:y_end+1,x_start:x_end+1])

    x = posterior(data,params,prior,11)
    x = np.pad(x.reshape(im_height-2*offset,im_width-2*offset),offset,'reflect')
    output_file = output_path + raw_image.split('/')[-1]
    cv2.imwrite(output_file,x)

    return x

def _get_gaussian_value(x, param,cl):
    mean = param[cl][0]
    std  = param[cl][1]
    return  1/(np.sqrt(2 * np.pi)*std) * np.e ** (-np.square(x-mean) / (2*np.square(std)))


def _get_prob(x, params, prior, patchsize, cl):
    prob = prior[cl]
    for i in range(3):
        prob = prob * _get_gaussian_value(x[i] , params[i],cl)
    return prob


def posterior(data, params, prior, patchsize):
    result = np.zeros(len(data))

    for i in range(len(data)):
        x = data[i]
        prob_0 = _get_prob(x,params,prior,patchsize,0)
        prob_1 = _get_prob(x,params,prior,patchsize,1)
        if prob_0 > prob_1:
            result[i] = 0
        else:
            result[i] = 1

    return result


if __name__=='__main__':

    if len(sys.argv) == 2:
        if not os.path.exists('output'):
            os.mkdir('output')
        sys.exit(main(sys.argv[1], sys.argv[2],'output/'))

    elif len(sys.argv) == 3 :
        if not os.path.exists(sys.argv[3]):
            os.mkdir(sys.argv[3])
        sys.exit(main(sys.argv[1], sys.argv[2],sys.argv[3]))
    else :
        print("argv 1 : raw_image, argv 2 : ground_truth image, argv 3 : output path for saving")
