# https://www.toptal.com/machine-learning/tensorflow-machine-learning-tutorial
# tensor board command: tensorboard --logdir=/tmp/tensorflowlogs
import tensorflow as tf

# we want to evaluate the function y = 5*x + 13

#In simple Python code, it would look like:

x = -2.0
y = 5*x + 13
print (y)

# in tensorflow
# only y is var, others are constant
x = tf.constant(-2.0, name="x", dtype=tf.float32)
a = tf.constant(5.0, name="a", dtype=tf.float32)
b = tf.constant(13.0, name="b", dtype=tf.float32)

y = tf.Variable(tf.add(tf.multiply(a, x), b))

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    print(session.run(y))

# create 1-D Tensor
import numpy as np
tensor_1d = np.array([1.45, -1, 0.2, 102.1])

print (tensor_1d)
#[   1.45   -1.      0.2   102.1 ]

print (tensor_1d[0])
#1.45

print (tensor_1d[2])
#0.2

print (tensor_1d.ndim)
#1

print (tensor_1d.shape)
#(4,)

print (tensor_1d.dtype)
#float64

# A NumPy array can be easily converted into a TensorFlow tensor
# with the auxiliary function convert_to_tensor,
# which helps developers convert Python objects to tensor objects.
#  This function accepts tensor objects, NumPy arrays, Python lists, and Python scalars.

tensor = tf.convert_to_tensor(tensor_1d, dtype=tf.float64)

with tf.Session() as session:
    print (session.run(tensor))
    print (session.run(tensor[0]))
    print (session.run(tensor[1]))


tensor_2d = np.array(np.random.rand(4, 4), dtype='float32')
tensor_2d_1 = np.array(np.random.rand(4, 4), dtype='float32')
tensor_2d_2 = np.array(np.random.rand(4, 4), dtype='float32')

m1 = tf.convert_to_tensor(tensor_2d)
m2 = tf.convert_to_tensor(tensor_2d_1)
m3 = tf.convert_to_tensor(tensor_2d_2)
mat_product = tf.matmul(m1, m2)
mat_sum = tf.add(m2, m3)
mat_det = tf.matrix_determinant(m3)

with tf.Session() as session:
    print(session.run(mat_product))
    print(session.run(mat_sum))
    print(session.run(mat_det))

# tf.add	x+y
# tf.subtract	x-y
# tf.multiply	x*y
# tf.div	x/y
# tf.mod	x % y
# tf.abs	|x|
# tf.negative	-x
# tf.sign	sign(x)
# tf.square	x*x
# tf.round	round(x)
# tf.sqrt	sqrt(x)
# tf.pow	x^y
# tf.exp	e^x
# tf.log	log(x)
# tf.maximum	max(x, y)
# tf.minimum	min(x, y)
# tf.cos	cos(x)
# tf.sin	sin(x)

# matrix operations
def convert(v, t=tf.float32):
    return tf.convert_to_tensor(v, dtype=t)

m1 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m2 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m3 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m4 = convert(np.array(np.random.rand(4, 4), dtype='float32'))
m5 = convert(np.array(np.random.rand(4, 4), dtype='float32'))

m_tranpose = tf.transpose(m1)
m_mul = tf.matmul(m1, m2)
m_det = tf.matrix_determinant(m3)
m_inv = tf.matrix_inverse(m4)
m_solve = tf.matrix_solve(m5, [[1], [1], [1], [1]])

with tf.Session() as session:
    print(session.run(m_tranpose))
    print(session.run(m_mul))
    print(session.run(m_inv))
    print(session.run(m_det))
    print(session.run(m_solve))


# reduction
# ensorFlow supports different kinds of reduction.
# Reduction is an operation that removes one or more dimensions
# from a tensor by performing certain operations across those dimensions.

x = convert(
    np.array(
        [
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9)
        ]), tf.int32)

print(x)
reduced_sum = tf.reduce_sum(x)
reduced_sum_horizontal = tf.reduce_sum(x,axis=1)
with tf.Session() as session:
    print(session.run(reduced_sum))
    print(session.run(reduced_sum_horizontal))
    # 45
    # [6 15 24]


import tensorflow as tf
x = tf.constant(1,name='x')
y = tf.Variable(x+9,name='y')
print(y)
# <tf.Variable 'y:0' shape=() dtype=int32_ref>
# y will be filled with data when compution done by session

#using placeholder
# data must be fed while run
a = tf.placeholder("int32")
b = tf.placeholder("int32")

y = tf.multiply(a,b)
# this function will return the result of the multiplication the input
# integers a and b.
# 5. Manage the execution flow, this means that we must build a session:
sess = tf.Session()
# 6. Visualize the results. We run our model on the variables a and b, feeding data
# into the data flow graph through the placeholders previously defined.
print(sess.run(y , feed_dict={a: 2, b: 5}))

# tensor board

a = tf.constant(10,name="a")
b = tf.constant(90,name="b")
y = tf.Variable(a+b*2, name="y")

model = tf.initialize_all_variables()

tf.merge_all_summaries = tf.summary.merge_all
tf.train.SummaryWriter = tf.summary.FileWriter

with tf.Session() as session:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/tmp/tensorflowlogs",session.graph)
    session.run(model)
    print(session.run(y))