#-*- coding: utf8 -*-
import os
import time
import multiprocessing
from multiprocessing import Process , current_process, Queue
import numpy as np
import pandas as pd
import cv2
import localBlur
from localBlur import LocalKurtosis, GradientHistogramSpan, LocalPowerSpectrumSlope

RAW_IMAGE_PATH = '../image/image/out_of_focus00{0:02d}.jpg'
GT_PATH = '../image/gt/out_of_focus00{0:02d}.png'
Q1_file = "Q1.csv"
Q2_file = "Q2.csv"
Q3_file = "Q3.csv"

def q1(img_nums, patchsize):

    for img_num in img_nums:
        print("{} 번째 q1 연산 시작".format(img_num))
        # 원본 이미지
        img = cv2.imread(RAW_IMAGE_PATH.format(img_num),cv2.IMREAD_GRAYSCALE)
        img = cv2.normalize(img.astype('double'),None,0.0,1.0,cv2.NORM_MINMAX)

        # ground Truth 이미지
        img_gt = cv2.imread(GT_PATH.format(img_num), cv2.IMREAD_GRAYSCALE)
        img_gt[img_gt==255] = 1

        # image 순서
        im_height, im_width = img.shape
        offset = int((patchsize -1)/2)
        x_start = offset+1; x_end = im_width-offset;
        y_start = offset+1; y_end = im_height-offset;
        datasize = (x_end-x_start+1)*(y_end-y_start+1)

        # 연산
        kurtosis = LocalKurtosis.LocalKurtosis(img, patchsize)

        data = np.zeros((datasize,4))
        data[:,0] = img_num
        data[:,1] = patchsize
        data[:,2] = np.ravel(kurtosis[y_start:y_end+1,x_start:x_end+1])
        data[:,3] = np.ravel(img_gt[y_start:y_end+1,x_start:x_end+1])

        data = pd.DataFrame(data, columns=['img_num','patchsize','kurtosis','gt'])

        if os.path.exists(Q1_file):
            before_df=pd.read_csv(Q1_file,encoding='utf-8',index_col=None)
            data=before_df.append(data)

        data.to_csv(Q1_file,encoding='utf-8',index =False)
        print("{} 번째 q1 연산 종료".format(img_num))


def q2(img_nums, patchsize):
    for img_num in img_nums:
        print("{} 번째 q2 연산 시작".format(img_num))
        # 원본 이미지
        img = cv2.imread(RAW_IMAGE_PATH.format(img_num),cv2.IMREAD_GRAYSCALE)
        img = cv2.normalize(img.astype('double'),None,0.0,1.0,cv2.NORM_MINMAX)

        # ground Truth 이미지
        img_gt = cv2.imread(GT_PATH.format(img_num), cv2.IMREAD_GRAYSCALE)
        img_gt[img_gt==255] = 1

        # image 순서
        im_height, im_width = img.shape
        offset = int((patchsize -1)/2)
        x_start = offset+1; x_end = im_width-offset;
        y_start = offset+1; y_end = im_height-offset;
        datasize = (x_end-x_start+1)*(y_end-y_start+1)

        # 연산
        histogram = GradientHistogramSpan.GradientHistogramSpan(img, patchsize)

        data = np.zeros((datasize,3))
        data[:,0] = img_num
        data[:,1] = patchsize
        data[:,2] = np.ravel(histogram[y_start:y_end+1,x_start:x_end+1])

        data = pd.DataFrame(data, columns=['img_num','patchsize','histogram'])

        if os.path.exists(Q2_file):
            before_df=pd.read_csv(Q2_file,encoding='utf-8',index_col=None)
            data=before_df.append(data)

        data.to_csv(Q2_file,encoding='utf-8',index =False)
        print("{} 번째 q2 연산 종료".format(img_num))


def q3(img_nums, patchsize):
    for img_num in img_nums:
        print("{} 번째 q3 연산 시작".format(img_num))
        # 원본 이미지
        img = cv2.imread(RAW_IMAGE_PATH.format(img_num),cv2.IMREAD_GRAYSCALE)
        img = cv2.normalize(img.astype('double'),None,0.0,1.0,cv2.NORM_MINMAX)

        # ground Truth 이미지
        img_gt = cv2.imread(GT_PATH.format(img_num), cv2.IMREAD_GRAYSCALE)
        img_gt[img_gt==255] = 1

        # image 순서
        im_height, im_width = img.shape
        offset = int((patchsize -1)/2)
        x_start = offset+1; x_end = im_width-offset;
        y_start = offset+1; y_end = im_height-offset;
        datasize = (x_end-x_start+1)*(y_end-y_start+1)

        # 연산
        powerspectrum = LocalPowerSpectrumSlope.LocalPowerSpectrumSlope(img, patchsize)

        data = np.zeros((datasize,3))
        data[:,0] = img_num
        data[:,1] = patchsize
        data[:,2] = np.ravel(powerspectrum[y_start:y_end+1,x_start:x_end+1])

        data = pd.DataFrame(data, columns=['img_num','patchsize','powerspectrum'])

        if os.path.exists(Q3_file):
            before_df=pd.read_csv(Q3_file,encoding='utf-8',index_col=None)
            data=before_df.append(data)

        data.to_csv(Q3_file,encoding='utf-8',index =False)
        print("{} 번째 q3 연산 종료".format(img_num))


if __name__ == '__main__' :

    img_nums = list(range(1,21))

    q1_pc = Process(target = q1, args= (img_nums,11))
    q2_pc = Process(target = q2, args = (img_nums,11))
    q3_pc = Process(target = q3, args= (img_nums,11))

    q1_pc.start()
    q2_pc.start()
    q3_pc.start()

    q1_pc.join()
    q2_pc.join()
    q3_pc.join()

