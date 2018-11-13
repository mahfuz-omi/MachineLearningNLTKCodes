# PIL (python imaging Library)
#  pip install Pillow

# https://auth0.com/blog/image-processing-in-python-with-pillow/
# https://nerdparadise.com/programming/pythonpil

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("digit.bmp", "r")

print(image)

# show image in external program
image.show()

# image = Image.open("png.png","r")
# print(image)

print(image.size)
#(28,28)

print(type(image))

print(list(image.getdata()))
# total 28*28 values
# [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]

image_array = np.array(list(image.getdata()))
print(image_array)

# [[255 255 255]
#  [255 255 255]
#  [255 255 255]
#  ...
#  [255 255 255]
#  [255 255 255]
#  [255 255 255]]

print(image_array.ndim)

print(image_array.shape)
# 2
# (784, 3)

plt.imshow(image_array)
plt.show()

print(image.format) # Output: BMP

# The pixel format used by the image. Typical values are “1”, “L”, “RGB”, or “CMYK.”
print(image.mode) # Output: RGB

# save image in different format
image.save('new_image.png')


# resizing image and save
new_image = image.resize((400, 400))
new_image.save('image_400.jpg')

image = Image.open("dog.jpg", "r")
print(image.format)
print(image.mode)
# JPEG
# RGB

# convert jpg to png
image.save('new_dog.png', 'PNG')

image = Image.open("new_dog.png", "r")
print(image.format)
print(image.mode)
# PNG
# RGB

image_array = np.array(list(image.getdata()))
print(image_array)


print(image_array.ndim)
print(image.size)

print(image_array.shape)
# 2
# (2880, 3840)
# (11059200, 3)

# it's a color image, but represented in 2d array
# it would be presented in 3d array
# rgb=color vs grayscale  https://www.quora.com/What-is-the-difference-between-grayscale-image-and-color-image

# convert to grayscale
image = image.convert('L')
print(image.mode)
print(image.size)

image_array = np.array(list(image.getdata()))
print(image_array)

# [70 74 71 ... 23 19 15]
# so, even the dimension has been changed
# 70 = .299*R+.587*G+.114*B

#image_array = image_array.reshape(image_array.shape[0],1)
print(image_array)
plt.imshow(image_array,cmap='gray')
plt.show()

import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical


#Importing MNIST Dataset



mnist = tf.keras.datasets.mnist
print(type(mnist))

(X_train, y_train),(X_test, y_test) = mnist.load_data()

print("X train",X_train)
print(type(X_train))
print("X train shape: ",X_train.shape)
print("X train dimension: ",X_train.ndim)
print("X train single image: ",X_train[100])

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# build image from numpy array of pixels
image = Image.fromarray(X_train[109])
print(image)
plt.imshow(image)
plt.show()