#-*- coding: utf8 -*-
import os
import time
import multiprocessing
from multiprocessing import Process , current_process, Queue, Lock
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

def q1(img_nums, patchsize,lock):

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

        with lock:
            if os.path.exists(Q1_file):
                before_df=pd.read_csv(Q1_file,encoding='utf-8',index_col=None)
                data=before_df.append(data)
            data.to_csv(Q1_file,encoding='utf-8',index =False)
        print("{} 번째 q1 연산 종료".format(img_num))


def q2(img_nums, patchsize,lock):
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

        data = np.zeros((datasize,4))
        data[:,0] = img_num
        data[:,1] = patchsize
        data[:,2] = np.ravel(histogram[y_start:y_end+1,x_start:x_end+1])
        data[:,3] = np.ravel(img_gt[y_start:y_end+1,x_start:x_end+1])


        data = pd.DataFrame(data, columns=['img_num','patchsize','histogram','gt'])

        with lock:
            if os.path.exists(Q2_file):
                before_df=pd.read_csv(Q2_file,encoding='utf-8',index_col=None)
                data=before_df.append(data)
            data.to_csv(Q2_file,encoding='utf-8',index =False)
        print("{} 번째 q2 연산 종료".format(img_num))


def q3(img_nums, patchsize,lock):
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

        data = np.zeros((datasize,4))
        data[:,0] = img_num
        data[:,1] = patchsize
        data[:,2] = np.ravel(powerspectrum[y_start:y_end+1,x_start:x_end+1])
        data[:,3] = np.ravel(img_gt[y_start:y_end+1,x_start:x_end+1])


        data = pd.DataFrame(data, columns=['img_num','patchsize','powerspectrum','gt'])

        with lock:
            if os.path.exists(Q3_file):
                before_df=pd.read_csv(Q3_file,encoding='utf-8',index_col=None)
                data=before_df.append(data)
            data.to_csv(Q3_file,encoding='utf-8',index =False)
        print("{} 번째 q3 연산 종료".format(img_num))


if __name__ == '__main__' :
    
    lock_q1 = Lock()
    lock_q2 = Lock()
    lock_q3 = Lock()

    img_nums   = list(range(16,21))
    img_nums_1 = [16]
    img_nums_2 = [17]
    img_nums_3 = [18]
    img_nums_4 = [20]

    q1_pc_1 = Process(target = q1, args= (img_nums,11,lock_q1))
    q2_pc_1 = Process(target = q2, args = (img_nums_1,11,lock_q2))
    q2_pc_2 = Process(target = q2, args = (img_nums_2,11,lock_q2))
    q2_pc_3 = Process(target = q2, args = (img_nums_3,11,lock_q2))
    q2_pc_4 = Process(target = q2, args = (img_nums_4,11,lock_q2))
    q3_pc_1 = Process(target = q3, args = ([16,17],11,lock_q3))
    q3_pc_2 = Process(target = q3, args = ([18,19,20],11,lock_q3))
    qc_pc_5 = Process(target = q2, args = ([19],11,lock_q2)) 

    q1_pc_1.start()
    q2_pc_1.start()
    q2_pc_2.start()
    q2_pc_3.start()
    q2_pc_4.start()
    q3_pc_1.start()
    q3_pc_2.start()
    qc_pc_5.start()

    q1_pc_1.join()
    q2_pc_1.join()
    q2_pc_2.join()
    q2_pc_3.join()
    q2_pc_4.join()
    q3_pc_1.join()
    q3_pc_2.join()
    qc_pc_5.join()

>>>>>>> 수정
