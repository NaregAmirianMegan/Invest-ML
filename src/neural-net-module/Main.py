import tensorflow as tf

class ANN():
	def __init__(self, hparams):
		self.hparams = hparams
		self._build_graph()

	def _build_graph(self):
		self.x = tf.placeholder(tf.float32, shape=(None, self.hparams['n_input']), name="X")
		self.y = tf.placeholder(tf.float32, shape=None, name="Y")

		hidden_layer_1 = tf.layers.dense(self.x, self.hparams['n_h1'], activation=tf.nn.relu, name="h1")
		hidden_layer_2 = tf.layers.dense(hidden_layer_1, self.hparams['n_h2'], activation=tf.nn.relu, name="h2")
		output_layer = tf.layers.dense(hidden_layer_2, self.hparams['n_output'], name="output")

		mse = tf.losses.mean_squared_error(self.y, output_layer)
		loss = tf.reduce_mean(mse, name="loss")