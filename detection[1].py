import cv2
import imutils
import numpy as np
import math
import pyaudio
import wave
import librosa
import json
import time


#
# time.sleep(5)
#
# t_end = time.time() + DURATION


# setting up audio module for recording sound
p = pyaudio.PyAudio()
CHANNELS = 1
RATE = 22050
CHUNK = 1024
stream = p.open(format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                frames_per_buffer=CHUNK)
frames = []
filename = "test.wav"

# expected_num_mfcc_vector_per_segment = math.ceil(SAMPLE_PER_TRACK/512)

# setting up video module for recording video
cap = cv2.VideoCapture(0)
Frame_out = np.zeros((500, 640, 3), np.uint8)

# Red color
low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])

# Green color
low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

# blue color
low_blue = np.array([94, 80, 2])
high_blue = np.array([126, 255, 255])

#Yellow color
low_yellow = np.array([20, 50, 50])
high_yellow = np.array([40, 255, 255])

mylist = {
    "x,y":[],
    "x1,y1":[],
    "mfcc":[],
    "labels":[]
}


#time.time() < t_end -> True
while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, low_red, high_red)
    green_mask = cv2.inRange(hsv, low_green, high_green)
    blue_mask = cv2.inRange(hsv,low_blue,high_blue)
    yellow_mask = cv2.inRange(hsv, low_yellow, high_yellow)
    left_mask = red_mask + green_mask
    right_mask = blue_mask + yellow_mask


    cnts = cv2.findContours(left_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts1 = cv2.findContours(right_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 250:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)

            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), -1)
            cv2.putText(frame, "center", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("frame", frame)


            data = stream.read(CHUNK)
            frames.append(data)




    print("area is ...", area)
    print("centroid is at ...", cx, cy)
    mylist["x,y"].append((cx,cy))
    cv2.circle(Frame_out, (cx, cy), 25, (0, 255, 255), 0)
    #cv2.imshow('animation', Frame_out)
    #cv2.circle(Frame_out, (cx, cy), 25, (0, 0, 0), 0)

    for c1 in cnts1:
        area1 = cv2.contourArea(c1)
        if area1 > 250:
            cv2.drawContours(frame, [c1], -1, (0, 255, 0), 3)
            M = cv2.moments(c1)
            cx1 = int(M["m10"] / M["m00"])
            cy1 = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx1, cy1), 7, (255, 255, 255), -1)
            cv2.putText(frame, "center", (cx1 - 20, cy1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("frame", frame)

    print("area1 is ...", area1)
    print("centroid1 is at ...", cx1, cy1)
    mylist["x1,y1"].append((cx1, cy1))
    cv2.circle(Frame_out, (cx, cy), 25, (0, 255, 255), 0)
    # cv2.imshow('animation', Frame_out)
    # cv2.circle(Frame_out, (cx, cy), 25, (0, 0, 0), 0)

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

stream.close()
p.terminate()

wf = wave.open(filename,"wb")
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
wf.setframerate(RATE)
wf.writeframes(b"".join(frames))
wf.close()


signal, sr = librosa.load('test.wav',RATE)


DURATION = librosa.get_duration(y=signal, sr=sr)
SAMPLE_PER_TRACK = RATE * DURATION


hop_length = 512
n_fft = 2048
n_mffc = 13
l = len(mylist["x,y"])



num_sample_per_segment = int(SAMPLE_PER_TRACK / l)
expected_num_mfcc_vector_per_segment = math.ceil(SAMPLE_PER_TRACK/ hop_length)

for s in range(l):
    start_sample = num_sample_per_segment * s
    finish_sample = start_sample + num_sample_per_segment

    mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                sr=sr,n_fft=n_fft, n_mfcc=n_mffc, hop_length=hop_length)
    mfcc = mfcc.T

    print(mfcc.shape)
    mylist["mfcc"].append(mfcc.tolist())
    mylist["labels"].append(1)

with open('dataset.txt','w') as outfile:
    json.dump(mylist,outfile,indent=4)


print(l,"=lenofXY",len(mylist["mfcc"]))