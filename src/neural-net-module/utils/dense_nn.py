import tensorflow as tf 
import numpy as np 

#################
### Functions ###
#################

def make_dense_nn(scope, n_inputs, n_outputs, h_layer_node_dict, initializer, activation_fn, loss_fn, lr=1e-4):
	"""
	Construct dense neural network using tf.layers.

	Args:
		scope - scope under which to define tf variables 
		n_inputs - number of inputs to network
		n_outputs - number of outputs of network
		h_layer_node_dict - dictionary of form: {"<layer_name>": <nodes_in_layer>, ...}
		initializer: function to initialize weights
		activation_fn: activation functions for layers
		loss_fn: loss function for network
		lr: learning rate of layers

	Returns: 
		x - input placeholder
		y - output placeholder
		output - output of network
		loss - loss of network
		train_op - training operation of network
	"""
	with tf.variable_scope(scope):
		x = tf.placeholder(tf.float32, [None, n_inputs], name="x")
		y = tf.placeholder(tf.float32, [None, n_outputs], name="y")
        
		prev_layer = x
		for layer_name, layer_nodes in h_layer_node_dict.items():
			prev_layer = tf.layers.dense(prev_layer, 	
            							 layer_nodes, 
            							 activation=activation_fn, 
            							 kernel_initializer=initializer, 
            							 name=layer_name)
        
		output = tf.layers.dense(prev_layer, n_outputs, name="outputs")
        
		loss = tf.reduce_mean(loss_fn(y, output))
		train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        
		return x, y, output, loss, train_op