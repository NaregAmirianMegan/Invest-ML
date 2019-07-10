import tensorflow as tf
import numpy as np 

def copy_weights(from_model_scope, to_model_scope, sess):
    for from_model_var, to_model_var in zip(tf.trainable_variables(from_model_scope), tf.trainable_variables(to_model_scope)):
        frm = from_model_var.eval(session=sess)
        to = to_model_var.eval(session=sess)
        np.copyto(to, frm)
        to_model_var.load(to, session=sess)