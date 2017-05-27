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

M_RAW_IMAGE_PATH = '../image/image/motion00{0:02d}.jpg'
M_GT_PATH = '../image/gt/motion00{0:02d}.png'


RAW_IMAGE_PATH = '../image/image/out_of_focus00{0:02d}.jpg'
GT_PATH = '../image/gt/out_of_focus00{0:02d}.png'

def multi(queue):
    while not queue.empty():
        raw_image, gt_image = queue.get()
        print("{}-multi start".format(raw_image))
        blurDetection.main(raw_image, gt_image, 'output/')


if __name__ == '__main__' :
    q = Queue(100)
    for i in range(1,21):
        q.put((RAW_IMAGE_PATH.format(i),GT_PATH.format(i)))
        q.put((M_RAW_IMAGE_PATH.format(i),M_GT_PATH.format(i)))
    if not os.path.exists('output'):
        os.mkdir('output')

    q1_pc = Process(target = multi, args = (q,))
    q2_pc = Process(target = multi, args = (q,))
    q3_pc = Process(target = multi, args = (q,))
    q4_pc = Process(target = multi, args = (q,))

    q1_pc_1 = Process(target = multi, args = (q,))
    q2_pc_1 = Process(target = multi, args = (q,))
    q3_pc_1 = Process(target = multi, args = (q,))
    q4_pc_1 = Process(target = multi, args = (q,))



    q1_pc.start()
    q2_pc.start()
    q3_pc.start()
    q4_pc.start()
    q1_pc_1.start()
    q2_pc_1.start()
    q3_pc_1.start()
    q4_pc_1.start()

    q1_pc.join()
    q2_pc.join()
    q3_pc.join()
    q4_pc.join()
    q1_pc_1.join()
    q2_pc_1.join()
    q3_pc_1.join()
    q4_pc_1.join()
