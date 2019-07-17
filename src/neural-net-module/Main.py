import tensorflow as tf
import numpy as np

# Local Imports
from utils.dense_nn import make_dense_nn

class Vanilla_NN:
	def __init__(self, n_inputs, n_outputs, h_layer_node_dict, initializer, activation_fn, loss_fn, lr, name, sess):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.h_layer_node_dict = h_layer_node_dict
		self.initializer = initializer
		self.activation_fn = activation_fn
		self.loss_fn = loss_fn
		self.lr = lr
		self.name = name
		self.sess = sess
		self._build_graph()
		self.sess.run(tf.global_variables_initializer())
		self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

	def _build_graph(self):
		self.x, self.y, self.output, self.loss, self.train_op = \
		make_dense_nn(scope=self.name, 
					  n_inputs=self.n_inputs, 
					  n_outputs=self.n_outputs, 
					  h_layer_node_dict=self.h_layer_node_dict, 
					  initializer=self.initializer, 
					  activation_fn=self.activation_fn, 
					  loss_fn=self.loss_fn, 
					  lr=self.lr)

	def _train_step(self, x, y):
		_, step_loss = sess.run([self.train_op, self.loss], feed_dict={self.x: x, self.y: y})
		return step_loss

	def predict(self, x):
		return self.sess.run(self.output, feed_dict={self.x: x})

	def train(self, x, y, train_steps, verbose=False, batching_fn=lambda x, y, batch_size: (x, y), batch_size=None, prog_freq=100):
		for step in range(train_steps):
			batch_x, batch_y = batching_fn(x, y, batch_size)
			step_loss = self._train_step(batch_x, batch_y)
			if verbose:
				if step % prog_freq:
					print("Loss:", step_loss)

if __name__ == '__main__':
	x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float32')
	y_data = np.array([[0], [1], [1], [0]], dtype='float32')

	

	with tf.Session() as sess:
		network = Vanilla_NN(2, 1, {'hidden_1': 10}, tf.contrib.layers.xavier_initializer(),
						 tf.nn.relu, tf.losses.mean_squared_error, 1e-3, 'network', sess)
		network.train(x_data, y_data, 10000, verbose=True)












