# -*- coding:utf-8 -*-

import cv2
import numpy as np
import time
from scipy import signal

window_name = 'Pulse Observer'
buffer_max_size = 350
min_hz = 0.83  # 50 BPM
max_hz = 3.33  # 200 BPM
graph_height = 200
min_frames = 30
show_fps = True  # Controls whether the FPS is displayed in top-left of GUI window.

# Lists for storing video frame data
values = []
times = []

# Creates the specified Butterworth filter and applies it.
# See:  http://scipy.github.io/old-wiki/pages/Cookbook/ButterworthBandpass
# def jiaozheng(roi):
#     B,G,R = cv2.split(roi)
#     maxR = np.max(R)
#     maxG = np.max(G)
#     maxB = np.max(B)
#     avgR = np.mean(R)
#     avgG = np.mean(G)
#     avgB = np.mean(B)
#     temp = [avgR/maxR,avgG/maxG,avgB/maxB]
#     k = np.min(temp)
#     R1 = k * R/avgR
#     G1 = k * G / avgR
#     B1 = k * B / avgR
#     I = cv2.merge([B1,G1,R1])
#     return I

# Draws the heart rate graph in the GUI window.
def draw_graph(signal, graph_width, graph_height):
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = 2
    scale_factor_y = 50
    midpoint_y = graph_height / 2
    for i in xrange(0, signal.shape[0] - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    return graph

# 在搜索到的框中找面积最大的
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

#累积多幅图像的绿色通道的值
def get_Gvalue(img,value):
    R, G, B = cv2.split(img)
    value1 = np.reshape(G, [1, -1])
    value1 = value1[0]
    value.append(value1.tolist())
    return value

# 计算图像的均值
def get_avgG(img):
    h, s, v = cv2.split(img)
    avg = np.mean(h)
    return avg
# 根据肤色计算
def fuse_avg(roi_color):
    Rsum = 0
    Gsum = 0
    Bsum = 0
    Hsum = 0
    count = 0
    rows, cols, channels = roi_color.shape
    # copy original image
    imgSkin = roi_color.copy()
    HSV = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)

    for r in range(rows):
        for c in range(cols):

            # get pixel value
            B = roi_color.item(r, c, 0)
            G = roi_color.item(r, c, 1)
            R = roi_color.item(r, c, 2)

            H = HSV.item(r,c,0)

            # non-skin area if skin equals 0, skin area otherwise
            skin = 0

            if (abs(R - G) > 15) and (R > G) and (R > B):
                if (R > 95) and (G > 50) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                    skin = 1
                    # print 'Condition 1 satisfied!'
                elif (R > 220) and (G > 210) and (B > 170):
                    skin = 1
                    # print 'Condition 2 satisfied!'

            if 0 == skin:
                imgSkin.itemset((r, c, 0), 0)
                imgSkin.itemset((r, c, 1), 0)
                imgSkin.itemset((r, c, 2), 0)
                # else:
                #     imgSkin.itemset((r, c, 0), 255)
                #     imgSkin.itemset((r, c, 1), 255)
                #     imgSkin.itemset((r, c, 2), 255)
            else:
                Rsum = Rsum+R
                Gsum = Gsum+G
                Bsum = Bsum+B
                Hsum = Hsum+H
                count = count+1

    count = float(count)
    Ravg = Rsum / count
    Gavg = Gsum / count
    Bavg = Bsum / count
    Havg = Hsum / count
    avg = (Ravg+Gavg+Bavg) / 3.0
    return Gavg

# def get_avgG(value):
#     row, col = np.shape(value)
#     avgs = []
#     for j in range(row):
#         img1 = value[j]
#         avg = np.mean(img1)
#         avgs.append(avg)
#     return np.array(avgs)


# 变化大的点的均值

def sliding_window_demean(signal, num_windows):
    window_size = int(round(len(signal) / num_windows))
    # print len(signal)
    # print window_size
    demeaned = np.zeros(signal.shape)
    for i in xrange(0, len(signal), window_size):
        if i + window_size > len(signal):
            window_size = len(signal) - i
        slice = signal[i:i + window_size]
        if slice.size == 0:
            print 'Empty Slice: size={0}, i={1}, window_size={2}'.format(signal.size, i, window_size)
            print slice
        demeaned[i:i + window_size] = slice - np.mean(slice)
    return demeaned

def linearSmooth5(input, N):
    output = np.zeros(input.shape)
    if ( N < 5 ):
        for i in range(N):
            output[i] = input[i]
    else:
        output[0] = ( 3.0 * input[0] + 2.0 * input[1] + input[2] - input[4] ) / 5.0;
        output[1] = ( 4.0 * input[0] + 3.0 * input[1] + 2 * input[2] + input[3] ) / 10.0;
        for i in range(2,N-2):
            output[i] = ( input[i - 2] + input[i - 1] + input[i] + input[i + 1] + input[i + 2] ) / 5.0;

        output[N - 2] = ( 4.0 * input[N - 1] + 3.0 * input[N - 2] + 2 * input[N - 3] + input[N - 4] ) / 10.0;
        output[N - 1] = ( 3.0 * input[N - 1] + 2.0 * input[N - 2] + input[N - 3] - input[N - 5] ) / 5.0;
    return output

# Main function.
def run_pulse_observer(webcam, window):
    last_bpm = 0
    frame_num = 0
    head_values = []
    nose_values = []

    # cv2.getWindowProperty() returns -1 when window is closed by user.
    while cv2.getWindowProperty(window, 0) == 0:
        r, frame = webcam.read()
        face_patterns = cv2.CascadeClassifier(
            '/Users/xiejinfan/anaconda/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')

        # print frame
        if r==True:
            frame_num += 1
            frame = cv2.GaussianBlur(frame, (5, 5), 1.5)
            view = np.array(frame)

            if frame_num == 1 or frame_num % 1 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_patterns.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(100, 100))
                face = maxbox(faces)

                # Make copy of frame before we draw on it.  We'll display the copy in the GUI.
                # The original frame will be used to compute heart rate.

                if (face is not None):
                    # Get the regions of interest.
                    center_y = face[1] + face[3] / 2
                    center_x = face[0] + face[2] / 2
                    fh_left = face[0] + int((center_x - face[0]) * 0.15)
                    # fh_right = face[0]+face[2] - int((center_x - face[0]) * 0.15)
                    # fh_top = int(face[1] * 0.88)
                    # fh_bottom = face[1]+face[3]
                    fh_right = face[0] + face[2] - int((center_x - face[0]) * 0.15)
                    fh_top = face[1]
                    fh_bottom = face[1] + face[3]
                    cv2.rectangle(view, (fh_left, fh_top), (fh_right, fh_bottom), color=(255, 0, 0))

                    # Slice out the regions of interest (ROI) and average them
                    fh_roi = frame[fh_top:fh_bottom, fh_left:fh_right]
                    # fB,fG,fR = cv2.split(fh_roi)
                    # fB_hist = cv2.equalizeHist(fB)
                    # fG_hist = cv2.equalizeHist(fG)
                    # fR_hist = cv2.equalizeHist(fR)
                    # fh_roi = cv2.merge([fB_hist,fG_hist,fR_hist])
                    # fh_roi1 = cv2.resize(fh_roi, (510, 510))

                    head_top = int(fh_top + (fh_bottom - fh_top) * 0.05)
                    head_bottom = int(fh_top + (fh_bottom - fh_top) * 0.2)
                    head_left = int(fh_left + (fh_right - fh_left) * 0.3)
                    head_right = int(fh_left + (fh_right - fh_left) * 0.7)
                    # head_top = 510 * 0.05
                    # head_bottom = int(510 * 0.25)
                    # head_left = int(510 * 0.27)
                    # head_right = int(510 * 0.7)
                    head_roi = view[head_top:head_bottom, head_left:head_right]
                    head_hsv = cv2.cvtColor(head_roi,cv2.COLOR_BGR2HSV)

                    # cv2.rectangle(view, (head_left1, head_top1), (head_right1, head_bottom1), color=(0, 255, 0))

                    nose_top = int(fh_top + (fh_bottom - fh_top) * 0.5)
                    nose_bottom = int(fh_top + (fh_bottom - fh_top) * 0.75)
                    nose_left = int(fh_left + (fh_right - fh_left) * 0.2)
                    nose_right = int(fh_left + (fh_right - fh_left) * 0.8)
                    # nose_top = int(510 * 0.5)
                    # nose_bottom = int(510 * 0.8)
                    # nose_left = int(510 * 0.15)
                    # nose_right = int(510 * 0.85)
                    nose_roi = view[nose_top:nose_bottom, nose_left:nose_right]
                    nose_hsv = cv2.cvtColor(nose_roi,cv2.COLOR_BGR2HSV)
                    # cv2.rectangle(view, (nose_left1, nose_top1), (nose_right1, nose_bottom1), color=(0, 255, 0))

                    # head_values = get_Gvalue(forehead_roi, head_values)
                    # nose_values = get_Gvalue(nose_roi, nose_values)

                    # avgs_head = get_avgG(head_hsv)
                    # avgs_nose = get_avgG(nose_roi)
                    # avgs = (avgs_head+avgs_nose)/2.0

                    avgs = fuse_avg(fh_roi)

                    # avgs = get_avgG(fh_roi)

                    values.append(avgs)
                    times.append(time.time())

            # Buffer is full, so pop the value off the top
            if len(times) > buffer_max_size:
                values.pop(0)
                times.pop(0)
            curr_buffer_size = len(times)

            graph_width = int(view.shape[1])

            if curr_buffer_size > min_frames:
                # print "valus:"+np.str(values)
                detrended = signal.detrend(np.array(values), type='linear')
                # print "detrended:"+np.str(detrended)
                demeaned = sliding_window_demean(detrended,15)
                # print "demended:"+np.str(demeaned)
                # demeaned = detrended

                b, a = signal.butter(5, 0.1, 'low')
                demeaned = signal.filtfilt(b, a, demeaned)
                # demeaned = linearSmooth5(demeaned, len(demeaned))
                filtered = demeaned
                # print filtered
                graph = draw_graph(filtered, graph_width, graph_height)
            else:
                graph = np.zeros((graph_height, graph_width, 3), np.uint8)

            view = np.vstack((view, graph))
            cv2.imshow(window, cv2.resize(view, (480, 700), interpolation=cv2.INTER_LINEAR))
        else:
            break

        key = cv2.waitKey(1)
        # Exit if user presses the escape key
        if key == 27:
            break

def main():

    webcam = cv2.VideoCapture('a.MOV')
    cv2.namedWindow(window_name)
    run_pulse_observer(webcam, window_name)
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
