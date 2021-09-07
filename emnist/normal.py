# import cv2
# from matplotlib import pyplot as plt
# image = cv2.imread('1.jpg')
# V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]

# from skimage.filters import threshold_local
# T = threshold_local(V, 15, offset=10, method="gaussian")
# thresh = (V > T).astype("uint8") * 255

# # convert black pixel of digits to white pixel
# thresh = cv2.bitwise_not(thresh)
# thresh = cv2.medianBlur(thresh, 5)

# thresh = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
# plt.imshow(thresh)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# img = cv.imread('4.png',0)
# img = cv.medianBlur(img,5)
# plt.imshow(img)
# plt.show()
# # ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)
# # plt.imshow(img)
# # plt.show()
# # th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
# #             cv.THRESH_BINARY,11,2)
# img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv.THRESH_BINARY,11,2)
# plt.imshow(img)
# plt.show()
# # titles = ['Original Image', 'Global Thresholding (v = 127)',
# #             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# # images = [img, th1, th2, th3]
# img = cv.resize(img, dsize=(28,28), interpolation=cv.INTER_CUBIC)
# plt.imshow(img)
# plt.show()
# img = np.array(img)
# t = np.copy(img)
# t = t / 255.0
# t = 1-t
# t = t.reshape(1,784)
# plt.imshow(img)
# plt.show()

# img = cv.imread('3.jpg',0)
# img = cv.medianBlur(img,5)
# plt.imshow(img)
# plt.show()
# img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#      cv.THRESH_BINARY,11,2)
# plt.imshow(img)
# plt.show()

image = cv.imread('3.jpg',0)
# height, width, depth = image.shape
plt.imshow(image)
plt.show()
#resizing the image to find spaces better
image2 = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
     cv.THRESH_BINARY,11,2)
# image = cv2.resize(image, dsize=(width*5,height*4), interpolation=cv2.INTER_CUBIC)

#grayscale
# gray = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                # cv2.THRESH_BINARY,11,2)
plt.imshow(image2)
plt.show()