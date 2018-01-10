#!/usr/bin/env python
# encoding: utf-8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

path = './data/'
def maxbox(img):
    max_box = 0
    if (img is not None) and len(img) > 1:
        for i in range(len(img)):
            temp = img[i]
            if temp[2] * temp[3] > max_box:
                max_box = temp[2] * temp[3]
                img_box = temp
    elif len(img)==1:
        img_box = img[0]
    else:
        img_box = None

    return img_box

def eyeposition(face):
    eye_sx = 0.16
    eye_sy = 0.26
    eye_sw = 0.30
    eye_sh = 0.28

    cols = len(face[0])
    rows = len(face)

    leftX = int(round(cols * eye_sx))
    topY = int(round(rows * eye_sy))
    widthX = int(round(cols * eye_sw))
    heightY = int(round(rows * eye_sh))
    rightX = int(round(cols * (1.0 - eye_sx - eye_sw)))  # start of right eye corner

    # Return the search windows to the caller, if desired
    searched_left_eye = [leftX, topY, widthX, heightY]
    searched_right_eye = [rightX, topY, widthX, heightY]
    return searched_left_eye,searched_right_eye

for filename in os.listdir(path):  # listdir的参数是文件夹的路径
    if os.path.splitext(filename)[1] == '.png':
        str = os.path.join(path, filename)
        img = cv2.imread(str)

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
        lefteye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30),flags = 0)

        face = maxbox(faces)
        # print faces

        if (face is not None):
            cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
            roi_gray = gray[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            roi_color = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=4, minSize=(20, 20))
            # for (ex, ey, ew, eh) in eyes:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            #
            eyesglass = eyeglasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.15, minNeighbors=4,
                                                                minSize=(20, 20))
            # for (ex, ey, ew, eh) in eyesglass:
            #     cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

            # searched_left_eye, searched_right_eye = eyeposition(face)
            eyeleft = lefteye_cascade.detectMultiScale(roi_gray, 1.1, minNeighbors=4, minSize=(20, 20))
            if len(eyeleft) > 0:
                for (ex, ey, ew, eh) in eyeleft:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

                eye_box = maxbox(eyeleft)
                eye_centerX = eye_box[0] + eye_box[2] / 2
                eye_centerY = eye_box[1] + eye_box[3] / 2

                eye = roi_color[eye_centerY-eye_box[3]/2:eye_centerY+eye_box[3]/2, eye_box[0]:eye_box[0] + eye_box[2]]
                eye_gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

                ret1,eye_bw = cv2.threshold(eye_gray, 50, 255, cv2.THRESH_BINARY)
                # contours, hierarchy = cv2.findContours(eye_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # print contours
                # 画出轮廓，-1,表示所有轮廓，画笔颜色为(0, 255, 0)，即Green，粗细为3
                # cv2.drawContours(eye_bw, contours, -1, (0, 255, 0), 3)
                # imS = cv2.resize(eye_bw, (480, 640))
                # cv2.imshow('img', imS)
                # cv2.waitKey(0)


                sumc = np.sum(255 - eye_bw, 0) / 255
                sumr = np.sum(255 - eye_bw, 1) / 255
                # print sumr

                eye_row_num = sum(sumr)

                # for i in range(len(sumr)):
                #     if i>=eye_centerX:
                #         eye_row_num += 1
                # print eye_row_num

                if eye_row_num>=100:
                    print "open eyes"
                else:
                    print "close eyes"


                plt.figure(1)
                plt.subplot(2, 2, 1), plt.imshow(eye_gray, 'gray'), plt.title('Original Image')
                plt.subplot(2, 2, 2), plt.imshow(eye_bw, 'gray'), plt.title('bw')
                plt.subplot(2, 2, 3), plt.plot(sumc), plt.title('sumc')
                plt.subplot(2, 2, 4), plt.plot(sumr), plt.title('sumr')
                plt.show()


        imS = cv2.resize(img, (480, 640))
        cv2.imshow('img'+filename, imS)

        cv2.waitKey(0)
cv2.destroyAllWindows()

        # print("faces:")
        # print(faces)
        # print("eyes:")
        # print(eyes)
        # print("eyesglass:")
        # print(eyesglass)
        # print("eyeleft:")
        # print(eyeleft)
        # print("********************")