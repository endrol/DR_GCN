import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import cmath

'''load the image sets'''
img_files = next(os.walk('/media/endrol/datafile/dataset/dr/idrid/B. Disease Grading/1. Original Images/a. Training Set'))[2]

# files_test = next(os.walk('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/'
#                           'idrid/B. Disease Grading/1. Original Images/b. Testing Set'))[2]
img_files.sort()
print(len(img_files))

'''deploy surf descriptor for each image'''
for img_fl in tqdm(img_files):
    if (img_fl.split('.')[-1] == 'jpg'):
        img = cv2.imread('/media/endrol/datafile/dataset/dr/idrid/B. Disease Grading/1. Original Images/a. Training Set/{}'.format(img_fl), cv2.IMREAD_COLOR)
        resized_img = cv2.resize(img, (672, 448), interpolation=cv2.INTER_CUBIC)
        # surf descriptor
        surf = cv2.xfeatures2d.SURF_create(1000)
        kp, des = surf.detectAndCompute(resized_img, None)

        if des is None:
            continue

        des = des[:20]

        np.save('/media/endrol/datafile/dataset/dr/idrid/B. Disease Grading/surf_des/{}'.format(img_fl.split('.')[0]), des)

#
#
# img = cv2.imread('data/test/IDRiD_001.jpg')
# surf = cv2.xfeatures2d.SURF_create(1000)
# sift = cv2.xfeatures2d.SIFT_create(1000)
#
# kp, des = sift.detectAndCompute(img, None)
# des = des[:20]
# print(des.shape)
#
# img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('data/test/new_oki_sift.jpg', img)