'''
state -> actor -> highest value action
state, action -> score 
'''
import tensorflow as tf 
import numpy as np 

class ActorCritic:
	def __init__(self, hparams):
		self.hparams = hparams

	def _create_actor_model(self, scope):
		with tf.variable_scope(scope):
			x = tf.placeholder(tf.float32, [None, self.hparams['n_inputs']])
			y = tf.placeholder(tf.float32, [None, self.hparams['n_outputs']])

			h1 = tf.layers.dense(x, self.hparams['hidden_layers']['n_h1'], activation=tf.nn.relu)
			h2 = tf.layers.dense(h1, self.hparams['hidden_layers']['n_h2'], activation=tf.nn.relu)
			h3 = tf.layers.dense(h2, self.hparams['hidden_layers']['n_h3'], activation=tf.nn.relu)
			output = tf.layers.dense(h2, self.hparams['n_outputs'])

	def _create_critic_model(self):