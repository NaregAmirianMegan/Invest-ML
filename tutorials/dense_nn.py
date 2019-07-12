import tensorflow as tf
import numpy as np 

def _make_dense_layer(prev_layer, n_inputs, n_nodes, activation_fn, layer_name):
    W = tf.Variable(tf.random.normal([n_inputs, n_nodes]), name=layer_name)
    b = tf.Variable(tf.random.normal([n_nodes]), name=layer_name)
    return activation_fn(tf.add(tf.matmul(prev_layer, W), b))

def _make_dense_nn(scope, n_inputs, n_outputs, h_layer_node_dict, loss_fn, lr=1e-4):
    with tf.variable_scope(scope):
        x = tf.placeholder(tf.float32, [None, n_inputs], name="x")
        y = tf.placeholder(tf.float32, [None, n_outputs], name="y")
        
        prev_layer = x
        prev_n_inputs = n_inputs
        for layer_name, layer_nodes in h_layer_node_dict.items():
            prev_layer = _make_dense_layer(prev_layer, prev_n_inputs, layer_nodes, tf.nn.relu, layer_name)
            prev_n_inputs = layer_nodes
        
        output = _make_dense_layer(prev_layer, prev_n_inputs, n_outputs, lambda x: x, "output")
        loss = tf.reduce_mean(loss_fn(y, output))
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        
        return x, y, output, loss, train_op

def _make_dense_nn_layers(scope, n_inputs, n_outputs, h_layer_node_dict, loss_fn, lr=1e-4):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        x = tf.placeholder(tf.float32, [None, n_inputs], name="x")
        y = tf.placeholder(tf.float32, [None, n_outputs], name="y")
        
        prev_layer = x
        for layer_name, layer_nodes in h_layer_node_dict.items():
            prev_layer = tf.layers.dense(prev_layer, layer_nodes, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name=layer_name)
        
        output = tf.layers.dense(prev_layer, n_outputs)
        
        loss = tf.reduce_mean(loss_fn(y, output))
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
        
        return x, y, output, loss, train_op

def _copy_weights(from_model_scope, to_model_scope, sess):
    for from_model_var, to_model_var in zip(tf.trainable_variables(from_model_scope), tf.trainable_variables(to_model_scope)):
        frm = from_model_var.eval(session=sess)
        to = to_model_var.eval(session=sess)
        np.copyto(to, frm)
        to_model_var.load(to, session=sess)