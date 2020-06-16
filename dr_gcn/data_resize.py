import numpy as np
import os
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import cmath
import time
import gc

'''this py file tries to change the image(from circle to square) and size'''

img_train = next(os.walk('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/'
                         'idrid/B. Disease Grading/1. Original Images/a. Training Set'))[2]

img_test = next(os.walk('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/'
                        'DataSet/idrid/B. Disease Grading/1. Original Images/b. Testing Set'))[2]

img_train.sort()
img_test.sort()

print(len(img_train))
print(len(img_test))

# counter = 0
for img_fl in tqdm(img_train):
    # if counter % 20 == 0:
    #     time.sleep(5)
    if (img_fl.split('.')[-1] == 'jpg'):
        img = cv2.imread('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/'
                         'idrid/B. Disease Grading/1. Original Images/a. Training Set/{}'.format(img_fl), cv2.IMREAD_COLOR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.medianBlur(gray_img, 5)
        '''using HoughCircles for circle detect'''
        circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 500,
                                   param1=50, param2=30, minRadius=100, maxRadius=0)
        del gray_img
        gc.collect()
        if circles is None:
            continue
        circles = np.uint16(np.around(circles))

        new_img = img[circles[0][0][1] - int(cmath.sqrt(2) / 2 * circles[0][0][2]): circles[0][0][1] + int(
            cmath.sqrt(2) / 2 * circles[0][0][2]),
                  circles[0][0][0] - int(cmath.sqrt(2) / 2 * circles[0][0][2]):circles[0][0][0] + int(
                      cmath.sqrt(2) / 2 * circles[0][0][2])]

        del img
        gc.collect()
        new_img = cv2.resize(new_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/'
                    'idrid/B. Disease Grading/resize_512/train/{}'.format(img_fl), new_img)

        del new_img
        gc.collect()

    # counter += 1


# counter = 0
# for img_fl in tqdm(img_test):
#     if counter % 20 == 0:
#         time.sleep(5)
#     if (img_fl.split('.')[-1] == 'jpg'):
#         img = cv2.imread('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/'
#                          'DataSet/idrid/B. Disease Grading/'
#                          '1. Original Images/b. Testing Set/{}'.format(img_fl), cv2.IMREAD_COLOR)
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray_img = cv2.medianBlur(gray_img, 5)
#         '''using HoughCircles for circle detect'''
#         circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 400,
#                                    param1=50, param2=40, minRadius=100, maxRadius=0)
#
#         if circles is None:
#             continue
#         circles = np.uint16(np.around(circles))
#         # print(circles)
#
#         new_img = img[circles[0][0][1] - int(cmath.sqrt(2) / 2 * circles[0][0][2]): circles[0][0][1] + int(
#             cmath.sqrt(2) / 2 * circles[0][0][2]),
#                   circles[0][0][0] - int(cmath.sqrt(2) / 2 * circles[0][0][2]):circles[0][0][0] + int(
#                       cmath.sqrt(2) / 2 * circles[0][0][2])]
#
#         new_img = cv2.resize(new_img, (512, 512), interpolation=cv2.INTER_CUBIC)
#         cv2.imwrite('/media/kamatalab/78cde73a-a99c-4bcc-b0af-7ba8c7da32f3/dan/Aiki_data/DataSet/'
#                     'idrid/B. Disease Grading/resize_512/test/{}'.format(img_fl), new_img)
#
#     counter += 1