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

image_path = 'input/image/{}'
output_path = 'output/{}/'
output_file = 'output/{}/{}_{}.jpg'

def multi(queue):
    while not queue.empty():
        image,prior,patchsize,out_file = queue.get()
        print("{}-multi start".format(image))
        image_file = image_path.format(image)
        blurDetection.experiment(image_file,prior,patchsize,out_file)

if __name__ == '__main__' :

    input_list = os.listdir('input/image/')
    patchsize_list = [11,13,15,17,19]
    prior_list = [0.1,0.3,0.5,0.7,0.9]

    q = Queue(600)

    # make path
    for image in input_list:
        image_name = image.split('.')[0]
        if not os.path.exists(output_path.format(image_name)):
            os.makedirs(output_path.format(image_name))

    # experiment parameter queue
    for image in input_list:
        for p in patchsize_list:
            for i in prior_list:
                image_name = image.split('.')[0]
                patchsize = p
                prior_set  = [i,1.0-i]
                output_name = output_file.format(image_name,str(patchsize),str(int(i*10)))
                q.put([image,[i,1-i],p,output_name])


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

