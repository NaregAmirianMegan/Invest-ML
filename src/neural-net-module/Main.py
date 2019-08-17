import tensorflow as tf
import numpy as np

# Local Imports
from utils.dense_nn import make_dense_nn
from utils.data_structures import CircularBuffer

#############################
### Feed forward networks ###
#############################

class Vanilla_NN:
	def __init__(self, n_inputs, n_outputs, h_layer_node_dict, initializer, activation_fn, loss_fn, lr, name, sess):
		"""
		Instantiates vanilla dense feed forward neural network.

		Args:
			n_inputs - number of input nodes
			n_outputs - number of ouptput nodes
			h_layer_node_dict - dictionary of form: {"<layer_name>": <nodes_in_layer>, ...}
			initializer - function to initialize weights
			activation_fn - activation functions for layers
			loss_fn - loss function for network
			lr - learning rate of layers
			name - name of scope in which to define graph
			sess - tensorflow session in which to run the graph

		Returns:
			Vanilla_NN object
		"""
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
		self.x, self.y, self.output, self.loss, self.train_op = make_dense_nn(scope=self.name, 
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
		"""
		Run prediction on x data.

		Args:
			x - data on which to run prediction

		Returns:
			Results of prediction
		"""
		return self.sess.run(self.output, feed_dict={self.x: x})

	def train(self, x, y, train_steps, verbose=False, batching_fn=lambda x, y, batch_size: (x, y), batch_size=None, prog_freq=100):
		"""
		Train model in sess on training data for train_steps steps.

		Args:
			x - X training data
			y - Y training data
			train_steps - number of training iterations
			verbose - displays loss as training progresses
			batching_fn - function that takes in x, y, and batch_size and returns batch_x and batch_y
			batch_size - size of batch
			prog_freq - how often to print loss

		Returns:
			None
		"""
		for step in range(train_steps):
			batch_x, batch_y = batching_fn(x, y, batch_size)
			step_loss = self._train_step(batch_x, batch_y)
			if verbose:
				if step % prog_freq:
					print("Loss:", step_loss)

#########################################
### Reinforcement Learning Algorithms ###
#########################################

class DQN:
	def __init__(self, n_inputs, n_outputs, h_layer_node_dict, initializer, 
				 activation_fn, loss_fn, lr, discount_rate, epsilon, e_decay, 
				 e_baseline, mem_size, name, sess):

	"""
		Instantiates deep q network.

		Args:
			n_inputs - number of input nodes
			n_outputs - number of ouptput nodes
			h_layer_node_dict - dictionary of form: {"<layer_name>": <nodes_in_layer>, ...}
			initializer - function to initialize weights
			activation_fn - activation functions for layers
			loss_fn - loss function for network
			lr - learning rate of layers
			discount_rate - fraction by which future rewards are discounted
			epsilon - probability of choosing random action
			e_decay - decrease probability of choosing random action at this rate
			e_baseline - baseline for epsilon
			mem_size - size of replay buffer
			name - name of scope in which to define graph
			sess - tensorflow session in which to run the graph

		Returns:
			DQN object
	"""

		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.h_layer_node_dict = h_layer_node_dict
		self.initializer = initializer
		self.activation_fn = activation_fn
		self.loss_fn = loss_fn
		self.lr = lr
		self.discount_rate = discount_rate
		self.epsilon = epsilon
		self.e_decay = e_decay
		self.e_baseline = e_baseline
		self.memory = CircularBuffer(mem_size)
		self.batch_size = batch_size
		self.name = name
		self.sess = sess
		self._build_graph()

		tf.trainable_variables(scope=name)

		self.update_target = [tf.trainable_variables(scope=self.name+"_"+"target-model").assign(
											tf.multiply(tf.trainable_variables(scope=self.name+"_"+"q-model"), self.tau) + 
											tf.multiply(tf.trainable_variables(scope=self.name+"_"+"q-model"), 1. - self.tau))
							  for i in range(len(tf.trainable_variables(scope=self.name+"_"+"target-model")))]


		self.sess.run(tf.global_variables_initializer())
		self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


		self.x, self.y, self.output_layer, self.loss, self.training_op = self._create_model('q-model')
		self.x_target, self.y_target, self.output_layer_target, self.loss_target, self.training_op_target = self._create_model('target-model')

	def _build_graph(self, scope_name):
		return make_dense_nn(scope=self.name+"_"+scope_name, 
					  		 n_inputs=self.n_inputs, 
					  		 n_outputs=self.n_outputs, 
					  		 h_layer_node_dict=self.h_layer_node_dict, 
					  		 initializer=self.initializer, 
					  		 activation_fn=self.activation_fn, 
					  		 loss_fn=self.loss_fn, 
					  		 lr=self.lr)
'''
	def _update_target_network(self, sess):
		_copy_weights("q-model", "target-model", sess)

	def predict(self, state, sess):
		return sess.run([self.output_layer], feed_dict={self.x: np.reshape(state, (1, self.hparams['n_inputs']))})

	def predict_batch(self, states, sess):
		return sess.run([self.output_layer_target], feed_dict={self.x_target: states})

	def get_action(self, state, sess):
		return np.argmax(self.predict(state, sess))

	def update_model(self, sess):
		if self.memory.currIndex <= self.batch_size and self.memory.full == False:
			return None

		batch = self.memory.random_sample(self.batch_size)

		states = [t[0] for t in batch]
		new_states = np.array([(np.zeros(self.hparams['n_inputs']) if t[3] is None else t[3]) for t in batch])
		q_vals = self.predict_batch(states, sess)[0]
		new_s_q_vals = self.predict_batch(new_states, sess)[0]
		x_batch = np.zeros((self.batch_size, self.hparams['n_inputs']))
		y_batch = np.zeros((self.batch_size, self.hparams['n_outputs']))
		for i, e in enumerate(batch):
			state, action, reward, new_state, done = e[0], e[1], e[2], e[3], e[4]
			current_q = q_vals[i]
			if done:
				current_q[action] = reward
			else:
				current_q[action] = reward + self.discount_rate*np.amax(new_s_q_vals[i])
			x_batch[i] = state
			y_batch[i] = current_q
		loss = self.train_batch(x_batch, y_batch, sess)
		return loss

	def train_batch(self, x_batch, y_batch, sess):
		_, curr_loss = sess.run([self.training_op, self.loss], feed_dict={self.x: x_batch, self.y: y_batch})
		return curr_loss

	def train(self, episodes, max_episode_length, sess, env, render_game=False):
		max_reward = 0
		avg_reward = 0
		m_avg_reward = []
		rewards = []

		for game in range(episodes):

			total_reward = 0

			env.reset()
			state = env.step(env.action_space.sample())[0]

			for step in range(max_episode_length):
				if render_game:
					env.render()
				if np.random.random() < self.epsilon:
					action = env.action_space.sample()
				else:
					action = self.get_action(state, sess).item()
				new_state, reward, done, info = env.step(action)
				total_reward += reward
				self.memory.append((state, action, reward, new_state, done))

				curr_loss = self.update_model(sess)

				if step % 20 == 0:
					self._update_target_network(sess)
								
				if done:
					break

				state = new_state
				self.epsilon = max(self.epsilon * self.e_decay, self.e_baseline)

			rewards.append(total_reward)
			m_avg_reward.append(total_reward)
			avg_reward = sum(rewards)/len(rewards)

			if total_reward > max_reward:
					max_reward = total_reward

			if game%50 == 0:
					print("=======================")
					print("GAME", game)
					moving_avg = sum(m_avg_reward)/len(m_avg_reward)
					print("Loss:", curr_loss, "M Avg. R:", moving_avg, "Max Reward:", max_reward, "Avg. R:", avg_reward, "Epsilon:", dqn.epsilon)
					m_avg_reward = []
					if moving_avg > 220:
						plt.plot(rewards)
						plt.ylabel('Rewards')
						plt.show()

		games = 10
		avg_score = self.eval(games, 1000, env)
		print("Model got an avg score of", avg_score, "over", games, "games.")
		    
		env.close()

	def eval(self, games, max_game_length, env):
		overall_reward = 0
		for game in range(games):
			env.reset()
			total_reward = 0
			state = env.step(env.action_space.sample())[0]

			for step in range(max_game_length):
				action = self.get_action(state, sess).item()
				new_state, reward, done, info = env.step(action)
				total_reward += reward

				if done:
					break
				state = new_state
			overall_reward += total_reward
		return overall_reward/games

if __name__ == '__main__':
	x_data = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype='float32')
	y_data = np.array([[0], [1], [1], [0]], dtype='float32')

	

	with tf.Session() as sess:
		network = Vanilla_NN(2, 1, {'hidden_1': 10}, tf.contrib.layers.xavier_initializer(),
						 tf.nn.relu, tf.losses.mean_squared_error, 1e-3, 'network', sess)
		network.train(x_data, y_data, 10000, verbose=True)












