# import os

# ### Set environment variable: echo ALPHA_VANTAGE_API_KEY=API_KEY
# API_KEY = os.environ['ALPHA_VANTAGE_API_KEY']

# ### alpha_vantage imports
# from alpha_vantage.techindicators import TechIndicators
# import matplotlib.pyplot as plt

# ti = TechIndicators(key=API_KEY, output_format='pandas')
# data, meta_data = ti.get_bbands(symbol='MSFT', interval='60min', time_period=60)
# data.plot()
# plt.title('BBbands indicator for  MSFT stock (60 min)')
# plt.show()


import tensorflow as tf 
import numpy as np

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float32')
outputs = np.array([[0], [1], [1], [0]], dtype='float32')

x = tf.placeholder(tf.float32, shape=(None, 2), name="x")
y = tf.placeholder(tf.float32, shape=None, name="y")

### hparams
n_h1 = 3
n_h2 = 2
n_output = 1

training_steps = 100000

### mid level implementation ###
l1 = tf.layers.dense(x, n_h1, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, n_h2, activation=tf.nn.relu)
out_l = tf.layers.dense(l2, n_output)

mse = tf.losses.mean_squared_error(outputs, out_l)
loss = tf.reduce_mean(mse)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
training_op = optimizer.minimize(loss)
### mid level (end) ###

### Low level implementation ###

# h1 = {'weights': tf.Variable(tf.random.normal([2, n_h1])), 
# 	  'biases': tf.Variable(tf.random.normal([n_h1]))}

# h2 = {'weights': tf.Variable(tf.random.normal([n_h1, n_h2])), 
# 	  'biases': tf.Variable(tf.random.normal([n_h2]))}

# output = {'weights': tf.Variable(tf.random.normal([n_h2, n_output])), 
# 	  	  'biases': tf.Variable(tf.random.normal([n_output]))}

# l1 = tf.add(tf.matmul(inputs, h1['weights']), h1['biases'])
# l2 = tf.add(tf.matmul(tf.nn.relu(l1), h2['weights']), h2['biases'])
# out_l = tf.add(tf.matmul(tf.nn.relu(l2), output['weights']), output['biases'])

# mse = tf.losses.mean_squared_error(outputs, out_l)
# loss = tf.reduce_mean(mse)
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# training_op = optimizer.minimize(loss)

### Low level (end) ###

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in range(training_steps):
		_, curr_loss = sess.run([training_op, loss], feed_dict={x: inputs, y: outputs})
		if step % 1000 == 0:
			print("Predictions:", np.round(sess.run(out_l, feed_dict={x: inputs}), decimals=0))
			print("Loss:", curr_loss)
