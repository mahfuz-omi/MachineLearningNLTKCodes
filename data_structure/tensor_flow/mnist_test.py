# https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
#Importing tensorflow
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical


#Importing MNIST Dataset
# it's grayscale image
# check mail drafy for nltk images for image structure



mnist = tf.keras.datasets.mnist
print(type(mnist))

(X_train, y_train),(X_test, y_test) = mnist.load_data()

# load_data returns 2d image
# there are 60000 images in training data. so, the shape of the output data will be 3d
print(type(X_train))

print(X_train.shape)
print(X_train.ndim)

# (60000, 28, 28)
# 3

print("x_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print(y_train)

# x_train shape: (60000, 28, 28) y_train shape: (60000,)
# [5 0 4 ... 5 6 8]

# one hot vectors is 0 in most dimension except for a single which
# denotes a value.

# Show one of the images from the training dataset
# imshow inputs numpy 2d array as image pixels
plt.imshow(X_train[5])
plt.show()

print(X_train[5])
# the 28*28 matrix (2D, numpy array of 2D)

# show corresponding digit
print(y_train[5])


# print test image
# pyplot here inputs only 2d image pixels
plt.imshow(X_test[5])
plt.show()

# print pic of 2
# 2

# from sklearn.preprocessing import LabelEncoder
# labelencoder = LabelEncoder()
# y_train = labelencoder.fit_transform(y_train)
# y_test = labelencoder.fit_transform(y_test)

# Reshaping the array to 4-dims so that it can work with the Keras API
# model takes input_shape as 3d image
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

print("x_train: ",X_train[10])

# Data normalization
# We then normalize the data dimensions so that they are of approximately the same scale.

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


# 1-hot encoding to target data
print("before to_categorical: ",y_train)
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print("after to_categorical: ",y_train)

# define the model
# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential()

# model takes input_shape as 3d image
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

#print(model.summary())

# We use model.compile() to configure the learning process
# before training the model. This is where you define the type of loss function,
# optimizer and the metrics evaluated by the model during training and testing.

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

# We will train the model with a batch_size of 64 and 10 epochs.

model.fit(X_train,
         y_train,epochs=10)
#
# # Evaluate the model on test set
# score = model.evaluate(X_test, y_test, verbose=0)
# # Print test accuracy
# print('\n', 'Test accuracy:', score[1])



# convert the 4d data to 3d data
print(X_test.shape)
X_test = X_test.reshape(10000,28,28)
print(X_test.shape)

print("test: ",X_test[500])
plt.imshow(X_test[500])
plt.show()

# show an image of 3

# convert the 3d data to 4d data
print(X_test.shape)
X_test = X_test.reshape(10000,28,28,1)
print(X_test.shape)

# predict this test data
y_pred = model.predict([[X_test[500]]],verbose=0)
print("prediction: ",y_pred)

print(y_pred.round(decimals=2))
# [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]]