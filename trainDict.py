import cv2
import os
import sys
import numpy as np

imagePath = 'data/'
dictSize = 100
print len(sys.argv)
if len(sys.argv) > 1: imagePath = sys.argv[1]
if len(sys.argv) > 2: dictSize = int(sys.argv[2])
#nfeatures = 500
#edgeThresh = 200

orb = cv2.ORB()
desAll = np.zeros((0,32), np.uint8)

for file in os.listdir(imagePath):
    if file.endswith('.jpg') or file.endswith('.JPG'):
        print imagePath + file
        newImg = cv2.imread(imagePath + file)
        gray = cv2.cvtColor(newImg, cv2.COLOR_BGR2GRAY)
        kp = orb.detect(gray, None)
        kp, des = orb.compute(gray, kp)
        print desAll.shape
        #print des.shape
        desAll = np.vstack((desAll, des))


desAll = np.float32(desAll)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(desAll, dictSize, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print center.shape

np.savetxt('dictionary.txt', center)
