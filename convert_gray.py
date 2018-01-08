import tensorflow as tf
import tflearn
from tflearn.data_utils import image_preloader
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import  cv2
MY_DATA='/home/kpranav1998/PycharmProjects/num_fingers/test.txt'
x_my_test,y_my_test=image_preloader(MY_DATA,mode='file',image_shape=(64,64),categorical_labels=True,normalize=True,grayscale=True)
image=cv2.imread('/home/kpranav1998/PycharmProjects/num_fingers/test_images/preyaa6.jpeg',cv2.IMREAD_GRAYSCALE)
image=cv2.resize(image,(300,300))
print(image.shape)
img = cv2.GaussianBlur(image, (7,7), 3)
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

plt.imshow(new.tolist(),cmap='gray')
plt.show(new.tolist())
cv2.imwrite( "/home/kpranav1998/PycharmProjects/num_fingers/test_images_gray/preyaa6_gray.jpg", new )

