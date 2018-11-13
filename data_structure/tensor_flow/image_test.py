import tensorflow as tf

# pip install pillow
import matplotlib.image as mp_image
filename = "dog.jpg"
input_image = mp_image.imread(filename)

#print(mp_image.shape)

# show image
import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

x = tf.Variable(input_image,name='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
# To perform the transpose of our matrix, use the transpose function of TensorFlow. This
# method performs a swap between the axes 0 and 1 of the input matrix, while the z axis is
# left unchanged:
    x = tf.transpose(x, perm=[1,0,2])
    session.run(model)
    result = session.run(x)

plt.imshow(result)
plt.show()