#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
import os
import blurDetection
from localBlur import LocalKurtosis, GradientHistogramSpan, LocalPowerSpectrumSlope


class demopractice:
    #demo 프로그램을 구동하는 데에 필요한 메소드들을 담은 class

    def __init__(self):
        self.image=cv2.imread('demo.jpg')
        # demo.jpg를 작업할 이미지로 setting

        self.copy = self.image
        self.image_name='demo_practice'
        cv2.namedWindow(self.image_name, cv2.WINDOW_NORMAL)

        cv2.imshow(self.image_name,self.image)
        cv2.setMouseCallback(self.image_name,self._click_and_blur)
        # _click_and_blur 함수를 이벤트 핸들러로 둠.
        # _click_and_blur : 드래그해서 blur할 구역을 설정해주는 함수


        while True:
            # 드래그를 한 후, enter or space를 누르면 이제
            # blur parameter, dirction과 magnitude를 결정할 수 있음

            key = cv2.waitKey(10)
            if (key == 13) or (key == 32) and len(refPt) == 2: # enter or space
                self.blurring_name='blurring'
                x0, y0 = self.refPt[0]
                x1, y1 = self.refPt[1]
                self.min_x = min(x0,x1); self.max_x = max(x0,x1)
                self.min_y = min(y0,y1); self.max_y = max(y0,y1)
                self.blur_image=self.image[self.min_y:self.max_y,self.min_x:self.max_x]
                cv2.namedWindow(self.blurring_name, cv2.WINDOW_NORMAL)
                cv2.createTrackbar('direction',self.blurring_name,-90,90,self.nothing)
                cv2.createTrackbar('magnitude',self.blurring_name,0,100,self.nothing)
                cv2.setTrackbarPos('direction',self.blurring_name,0)
                cv2.setTrackbarPos('magnitude',self.blurring_name,0)
                cv2.imshow(self.blurring_name,self.blur_image)
                break


        while True:
            # blur parameter를 결정하는 코드
            key = cv2.waitKey(10)
            direction = cv2.getTrackbarPos('direction', self.blurring_name)
            magnitude = cv2.getTrackbarPos('magnitude', self.blurring_name)

            # --- FIXME ---
            # direction & magnitude 값을 이용해서 이미지를 blur화시켜야 함.
            # ex)
            # self.blur_image = blurring(self.blur_image, direction, magnitude)

            cv2.imshow(self.blurring_name, self.blur_image)
            if (key==ord('y')):
                # y를 누르면 지나감
                cv2.destroyWindow(self.blurring_name)
                break

        self.image[self.min_y:self.max_y,self.min_x:self.max_x] = self.blur_image
        # 위에서 self.blur_image를 바꾸어준 것을 다시 self.image에 넣음
        cv2.imshow(self.image_name,self.image)

        self.mask = blurDetection.demo(self.image)
        cv2.namedWindow('blur-mask', cv2.WINDOW_NORMAL)
        cv2.imshow('blur-mask',self.mask)

        # ---- FIXME ---
        # self.mask를 이용해서
        # blur_image를 restoration 시켜주어야 함.
        # ex)
        # self.restoration_image = restoration(self.image,self.mask)

        cv2.namedWindow('image restoration', cv2.WINDOW_NORMAL)
        cv2.imshow('image restoration', self.restoration_image)

        while True:
        # 종료 코드, enter or space 누르면 끝.
            key = cv2.waitKey(10)
            if (key == 13) or (key == 32):
                cv2.destroyAllWindows()
                break


    def nothing(*arg):
        pass


    def _click_and_blur(self,event, x, y, flags, param):
        # _click_and_blur : 드래그해서 blur할 구역을 설정해주는 함수
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image = self.copy
            self.refPt = [(x,y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self.refPt.append((x,y))
            if len(self.refPt) == 2:
                cv2.rectangle(self.image, self.refPt[0], self.refPt[1], (20,150,10), 2)
                cv2.imshow(self.image_name, self.image)


demopractice() # 실제 실행 코드