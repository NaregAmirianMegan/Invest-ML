import tensorflow as tf
import numpy as np

class ANN():
	def __init__(self, hparams):
		self.hparams = hparams
		self._build_graph(hparams)

	def _build_graph(self, hparams):
		self.x = tf.placeholder(tf.float32, shape=(None, hparams['n_input']), name="X")
		self.y = tf.placeholder(tf.float32, shape=None, name="Y")

		hidden_layer_1 = tf.layers.dense(self.x, hparams['n_h1'], activation=tf.nn.relu, name="h1")
		hidden_layer_2 = tf.layers.dense(hidden_layer_1, hparams['n_h2'], activation=tf.nn.relu, name="h2")
		self.output_layer = tf.layers.dense(hidden_layer_2, hparams['n_output'], name="output")

		mse = tf.losses.mean_squared_error(self.y, self.output_layer)
		self.loss = tf.reduce_mean(mse, name="loss")
		optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams['lr'])
		self.training_op = optimizer.minimize(self.loss)

	def _train_step(self, x, y, sess):
		_, step_loss = sess.run([self.training_op, self.loss], feed_dict={self.x: x, self.y: y})
		return step_loss

	def _get_batch(self, x, y, size=0, no_batch=False):
		if no_batch:
			return x, y

	def train(self, x, y, sess, verbose=False):
		for step in range(self.hparams['training_steps']):
			batch_x, batch_y = self._get_batch(x, y, no_batch=True)
			step_loss = self._train_step(batch_x, batch_y, sess)
			if verbose:
				if step % (self.hparams['training_steps']/1000):
					print("Loss:", step_loss)
					print(self.predict(batch_x, sess))

	def predict(self, x_data, sess):
		return sess.run(self.output_layer, feed_dict={self.x: x_data})

	def eval(self, ):

class RNN():
	def __init__(self, hparams):
		

if __name__ == '__main__':
	x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float32')
	y_data = np.array([[0], [1], [1], [0]], dtype='float32')

	hparams = {'n_input': 2, 'n_h1': 10, 'n_h2': 5, 'n_output': 1, 'lr': 0.0001, 'training_steps': 10000}
	ann = ANN(hparams)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		ann.train(x_data, y_data, sess, verbose=True)