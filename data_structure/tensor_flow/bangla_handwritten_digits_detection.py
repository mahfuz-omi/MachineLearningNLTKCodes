# # from PIL import Image
# # import matplotlib.pyplot as plt
# # import numpy as np
# #
# #
# # # test image
# # image = Image.open("bangla_digits/one/one1541424040167.png", "r")
# #
# # print(image)
# # # <PIL.PngImagePlugin.PngImageFile image mode=RGBA size=40x40 at 0x22F693C8C50>
# #
# # print(image.mode)
# # # RGBA
# #
# # print(list(image.getdata()))
# # # [(0, 0, 0, 0), (0, 0, 0, 3), (4, 4, 4, 121), (4, 4, 4, 127)
# #
# # image_array = np.array(list(image.getdata()))
# # print(image_array)
# #
# # # [[  0   0   0   0]
# # #  [  0   0   0   3]
# # #  [  4   4   4 121]
# # #  ...
# # #  [  0   0   0   0]
# # #  [  0   0   0   0]
# # #  [  0   0   0   0]]
# #
# # # convert to GrayScale
# # image = image.convert('L')
# #
# #
# # image_array = np.array(list(image.getdata()))
# # print(image_array)
# #
# # # [[0 0 0]
# # #  [0 0 0]
# # #  [4 4 4]
# # #  ...
# # #  [0 0 0]
# # #  [0 0 0]
# # #  [0 0 0]]
#
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from keras.utils import to_categorical
#
#
# #Importing MNIST Dataset
#
#
#
# # mnist = tf.keras.datasets.mnist
# # print(type(mnist))
# #
# # (X_train, y_train),(X_test, y_test) = mnist.load_data()
# #
# # print("X train",X_train)
# # print(type(X_train))
# # print("X train shape: ",X_train.shape)
# # print("X train dimension: ",X_train.ndim)
# # print("X train single image: ",X_train[100])
#
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.nan)
# #
# # image = Image.fromarray(X_train[109])
# # print(image)
# # plt.imshow(image)
# # plt.show()
#
image = Image.open("bangla_digits/one/one1541424040167.png", "r")
print(image)
plt.imshow(image)
plt.show()


image_array = np.array(image.getdata())
print(image_array)
#
image = image.convert('RGB')
print(image)
plt.imshow(image,cmap='gray')
plt.show()

image = image.convert('L')
print(image)
plt.imshow(image,cmap='gray')
plt.show()
#
image_array = np.array(image.getdata())
print(image_array)
#
image_array = image_array.reshape(40,40)
print(image_array)

image = Image.fromarray(image_array)
plt.imshow(image)
plt.show()
#
# print(image_array.ndim)
# print(image_array.shape)

# print(image_array.ndim)
#
# image_array = image_array.reshape(40,40,1)
# print(image_array)
#
# print(image_array.ndim)

# import glob
# import os
#
# imageList = []
# labelList = []
#
# for filepath in glob.iglob('bangla_digits/one/*.png'):
#     filename = os.path.basename(filepath)
#     image = Image.open("bangla_digits/one/"+filename,"r")
#     print(image.mode)
#     image = image.convert('L')
#     image_array = np.array(image.getdata())
#     image_array = image_array.reshape(40, 40)
#     imageList.append(image_array)
#     labelList.append(1)
#
# for filepath in glob.iglob('bangla_digits/two/*.png'):
#     filename = os.path.basename(filepath)
#     image = Image.open("bangla_digits/two/"+filename,"r")
#     print(image.mode)
#     image = image.convert('L')
#     image_array = np.array(image.getdata())
#     image_array = image_array.reshape(40, 40)
#     imageList.append(image_array)
#     labelList.append(2)
#
# for filepath in glob.iglob('bangla_digits/three/*.png'):
#     filename = os.path.basename(filepath)
#     image = Image.open("bangla_digits/three/"+filename,"r")
#     print(image.mode)
#     image = image.convert('L')
#     image_array = np.array(image.getdata())
#     image_array = image_array.reshape(40, 40)
#     imageList.append(image_array)
#     labelList.append(3)
#
# for filepath in glob.iglob('bangla_digits/four/*.png'):
#     filename = os.path.basename(filepath)
#     image = Image.open("bangla_digits/four/"+filename,"r")
#     print(image.mode)
#     image = image.convert('L')
#     image_array = np.array(image.getdata())
#     image_array = image_array.reshape(40, 40)
#     imageList.append(image_array)
#     labelList.append(4)
#
#
# print(np.array(imageList))
# print(imageList.__len__())
# print(np.array(labelList))
# print(labelList.__len__())
#
# X_train = np.array(imageList)
# y_train = np.array(labelList)
#
# image = Image.fromarray(X_train[50])
# plt.imshow(image)
# plt.show()
#
#
#
#
#
#
