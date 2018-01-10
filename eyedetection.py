#!/usr/bin/env python
# encoding: utf-8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 选取最大框作为人脸
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

# 人为固定的眼睛范围区
def eyeposition(face):
    eye_sx = 0.16
    eye_sy = 0.26
    eye_sw = 0.30
    eye_sh = 0.28

    cols = face[2]
    rows = face[3]

    leftX = int(round(cols * eye_sx))
    topY = int(round(rows * eye_sy))
    widthX = int(round(cols * eye_sw))
    heightY = int(round(rows * eye_sh))
    rightX = int(round(cols * (1.0 - eye_sx - eye_sw)))  # start of right eye corner

    # Return the search windows to the caller, if desired
    searched_left_eye = [leftX, topY, widthX, heightY]
    searched_right_eye = [rightX, topY, widthX, heightY]
    # cv2.rectangle(roi_color, (searched_left_eye[0], searched_left_eye[1]),
    #               (searched_left_eye[0] + searched_left_eye[2], searched_left_eye[1] + searched_left_eye[3]), (255, 0, 255))
    # cv2.rectangle(roi_color, (searched_right_eye[0], searched_right_eye[1]),
    #               (searched_right_eye[0] + searched_right_eye[2],searched_right_eye[1] + searched_right_eye[3]),
    #               (255, 0, 255))
    return searched_left_eye, searched_right_eye

# 各分类器检测眼睛
def eye_detection(roi_gray, face):
    # 最终存放检测到的眼睛框
    lefteye_box = None
    righteye_box = None
    eye_box = None
    # 固定的眼睛区
    searched_left_eye, searched_right_eye = eyeposition(face)
    # 眼睛分类器检测
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    lefteye_cascade = cv2.CascadeClassifier('haarcascade_lefteye_2splits.xml')
    righteye_cascade = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')

    m, n = np.shape(roi_gray)
    leftface_gray = roi_gray[0:m, n / 2:n]      # 半脸检测，左眼对应图片右区
    rightface_gray = roi_gray[0:m, 0:n / 2]

    eyeleft = lefteye_cascade.detectMultiScale(leftface_gray, 1.1, minNeighbors=4, minSize=(20, 20))
    eyeright = righteye_cascade.detectMultiScale(rightface_gray, 1.1, minNeighbors=4, minSize=(20, 20))
    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    eyesglass = eyeglasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4,minSize=(20, 20))
    # 检测左眼
    if len(eyeleft) > 0:
        lefteye_box = maxbox(eyeleft)
        lefteye_box[0] = lefteye_box[0]+n/2
        print(lefteye_box[1] + lefteye_box[3]/2 - (searched_left_eye[1]+searched_left_eye[3]/2))
        # 眼应位于上半脸，且跟固定眼区中心相差小于20，否则清空
        if face[1]+lefteye_box[1] + lefteye_box[3] / 2 >= face[1] + face[3] / 2:
            lefteye_box = None
        elif abs(lefteye_box[1] + lefteye_box[3]/2 - (searched_left_eye[1]+searched_left_eye[3]/2)) > 30:
                lefteye_box = None
    # 检测右眼
    if len(eyeright) > 0:
        righteye_box = maxbox(eyeright)
        # 眼应位于上半脸，且跟固定眼区中心相差小于20，否则清空
        if face[1]+righteye_box[1] + righteye_box[3] / 2 >= face[1] + face[3] / 2:
            righteye_box = None
        elif abs(righteye_box[1] + righteye_box[3]/2 - (searched_right_eye[1]+searched_right_eye[3]/2)) > 30 :
            righteye_box = None
    # 如果左右眼都检测失败，再用eye和eyeglass尝试
    if (lefteye_box is None) and (righteye_box is None):
        if len(eyes) > 0:
            eye_box = maxbox(eyes)
            if face[1]+eye_box[1] + eye_box[3] / 2 >= face[1] + face[3] / 2:
                eye_box = None
            elif abs(eye_box[1] + eye_box[3]/2 - (searched_right_eye[1]+searched_right_eye[3]/2)) > 30 :
                eye_box = None

        if eye_box is None:
            if len(eyesglass) > 0:
                eye_box = maxbox(eyesglass)
                if face[1]+eye_box[1] + eye_box[3] / 2 >= face[1] + face[3] / 2:
                    eye_box = None
                elif abs(eye_box[1] + eye_box[3]/2 - (searched_right_eye[1]+searched_right_eye[3]/2)) > 30:
                    eye_box = None

    return lefteye_box,righteye_box,eye_box

def eyeDect(webcam, window):

    while cv2.getWindowProperty(window, 0) == 0:
        r, img = webcam.read()
        view = np.array(img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # 灰度图

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30), flags=0)

        face = maxbox(faces)  # 检测到的最大的脸框
        # print faces

        if (face is not None):
            cv2.rectangle(img, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (255, 0, 0), 2)
            # 得到感兴趣的脸区域
            roi_gray = gray[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            roi_color = img[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            # 检测返回的眼睛框
            lefteye_box, righteye_box, eye_box = eye_detection(roi_gray, face)
            if lefteye_box is not None:
                cv2.rectangle(roi_color, (lefteye_box[0], lefteye_box[1]),
                              (lefteye_box[0] + lefteye_box[2], lefteye_box[1] + lefteye_box[3]),(0, 0, 255))
            if righteye_box is not None:
                cv2.rectangle(roi_color, (righteye_box[0], righteye_box[1]),
                              (righteye_box[0] + righteye_box[2], righteye_box[1] + righteye_box[3]), (0, 255, 0))
            elif (eye_box is not None):
                cv2.rectangle(roi_color, (eye_box[0], eye_box[1]), (eye_box[0] + eye_box[2], eye_box[1] + eye_box[3]),
                              (0, 255, 255))


def main():
    webcam = cv2.VideoCapture(0)
    cv2.namedWindow('eye_detection')
    eyeDect(webcam, 'eye_detection')
    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()