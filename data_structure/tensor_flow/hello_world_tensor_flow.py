import tensorflow as tf

# create constant
constant = tf.constant("hello world !!")

# create session to run
session = tf.Session()

# run
print(session.run(constant))

# close session
session.close()