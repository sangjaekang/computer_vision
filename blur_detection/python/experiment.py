import os
import time
import multiprocessing
from multiprocessing import Process , current_process, Queue
import numpy as np
import pandas as pd
import cv2
import localBlur
import blurDetection
from localBlur import LocalKurtosis, GradientHistogramSpan, LocalPowerSpectrumSlope

image_path = 'input/motion/{}'
output_path = 'output/{}/'
output_file = 'output/{}/{}_{}.jpg'

def multi(queue):
    while not queue.empty():
        image,prior,patchsize,out_path = queue.get()
        print("{}-multi start".format(image))
        image_file = image_path.format(image)
        blurDetection.main(image_file,prior,patchsize,out_path)

if __name__ == '__main__' :

    input_list = os.listdir('input/motion/')

    q = Queue(300)

    # make path
    if not os.path.exists('output/kurtosis'):
        os.makedirs('output/kurtosis')
    if not os.path.exists('output/histogram'):
        os.makedirs('output/histogram')
    if not os.path.exists('output/power'):
        os.makedirs('output/power')
    if not os.path.exists('output/naive'):
        os.makedirs('output/naive')

    # experiment parameter queue
    for image in input_list:
        image_name = image.split('.')[0]
        patchsize = 11
        prior_set  = [0.6630,0.3370]
        q.put([image,prior_set,patchsize,'output/'])


    q1_pc = Process(target = multi, args = (q,))
    q2_pc = Process(target = multi, args = (q,))
    q3_pc = Process(target = multi, args = (q,))
    q4_pc = Process(target = multi, args = (q,))
    q1_pc_1 = Process(target = multi, args = (q,))
    q2_pc_1 = Process(target = multi, args = (q,))
    q3_pc_1 = Process(target = multi, args = (q,))
    q4_pc_1 = Process(target = multi, args = (q,))
    q1_pc_2 = Process(target = multi, args = (q,))
    q2_pc_2 = Process(target = multi, args = (q,))
    q3_pc_2 = Process(target = multi, args = (q,))
    q4_pc_2 = Process(target = multi, args = (q,))
    q1_pc_3 = Process(target = multi, args = (q,))
    q2_pc_3 = Process(target = multi, args = (q,))
    q3_pc_3 = Process(target = multi, args = (q,))
    q4_pc_3 = Process(target = multi, args = (q,))


    q1_pc.start()
    q2_pc.start()
    q3_pc.start()
    q4_pc.start()
    q1_pc_1.start()
    q2_pc_1.start()
    q3_pc_1.start()
    q4_pc_1.start()
    q1_pc_2.start()
    q2_pc_2.start()
    q3_pc_2.start()
    q4_pc_2.start()
    q1_pc_3.start()
    q2_pc_3.start()
    q3_pc_3.start()
    q4_pc_3.start()

    q1_pc.join()
    q2_pc.join()
    q3_pc.join()
    q4_pc.join()
    q1_pc_1.join()
    q2_pc_1.join()
    q3_pc_1.join()
    q4_pc_1.join()
    q1_pc_2.join()
    q2_pc_2.join()
    q3_pc_2.join()
    q4_pc_2.join()
    q1_pc_3.join()
    q2_pc_3.join()
    q3_pc_3.join()
    q4_pc_3.join()

