import gym, random
import tensorflow as tf
import numpy as np



class CircularBuffer:
	def __init__(self, size):
		self.data = [None] * size
		self.size = size
		self.currIndex = 0
		self.full = False

	def append(self, element):
		if(self.currIndex == self.size):
			self.full = True
			self.currIndex = 0
		self.data[self.currIndex] = element
		self.currIndex += 1

	def random_sample(self, batch_size):
		sample = [None] * batch_size
		for x in range(batch_size):
			if(self.full):
				sample[x] = self.data[random.randint(0, self.size-1)]
			else:
				sample[x] = self.data[random.randint(0, self.currIndex-1)]
		return sample

class DQN:
	def __init__(self, hparams):
		self.hparams = hparams
		self.lr = hparams['lr']
		self.discount_rate = hparams['discount_rate']

		self.epsilon = hparams['epsilon']
		self.e_decay = hparams['e_decay']
		self.e_baseline = hparams['e_baseline']

		self.memory = CircularBuffer(25000)

		self.batch_size = hparams['batch_size']

		self.x, self.y, self.output_layer, self.loss, self.training_op = self.create_model(hparams, 'q-model')
		self.x_target, self.y_target, self.output_layer_target, self.loss_target, self.training_op_target = self.create_model(hparams, 'target-model')

	def create_model(self, hparams, scope):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			x = tf.placeholder(tf.float32, shape=[None, hparams['n_state_nodes']])
			y = tf.placeholder(tf.float32, shape=[None, hparams['n_actions']])

			hidden_layer_1 = tf.layers.dense(x, hparams['n_h1'], activation=tf.nn.relu)
			hidden_layer_2 = tf.layers.dense(hidden_layer_1, hparams['n_h2'], activation=tf.nn.relu)
			hidden_layer_3 = tf.layers.dense(hidden_layer_2, hparams['n_h3'], activation=tf.nn.relu)
			output_layer = tf.layers.dense(hidden_layer_3, hparams['n_actions'])

			mse = tf.losses.mean_squared_error(y, output_layer)
			loss = tf.reduce_mean(mse)
			optimizer = tf.train.AdamOptimizer(learning_rate=hparams['lr'])
			training_op = optimizer.minimize(loss)

			return x, y, output_layer, loss, training_op

	def update_target_model(self, sess):
		update_weights = [tf.assign(new, old) for (new, old) in zip(tf.trainable_variables('q-model'), tf.trainable_variables('target-model'))]
		sess.run([update_weights])

	def predict(self, state, sess):
		return sess.run([self.output_layer], feed_dict={self.x: np.reshape(state, (1, self.hparams['n_state_nodes']))})

	# Use target network for predictions
	def predict_batch(self, states, sess):
		return sess.run([self.output_layer_target], feed_dict={self.x_target: states})

	def get_action(self, state, sess):
		return np.argmax(self.predict(state, sess))

	def update_model(self, sess):
		if self.memory.currIndex <= self.batch_size and self.memory.full == False:
			return None

		batch = self.memory.random_sample(self.batch_size)

		states = [t[0] for t in batch]
		new_states = np.array([(np.zeros(self.hparams['n_state_nodes']) if t[3] is None else t[3]) for t in batch])
		q_vals = self.predict_batch(states, sess)[0]
		new_s_q_vals = self.predict_batch(new_states, sess)[0]
		x_batch = np.zeros((self.batch_size, self.hparams['n_state_nodes']))
		y_batch = np.zeros((self.batch_size, self.hparams['n_actions']))
		for i, e in enumerate(batch):
			state, action, reward, new_state = e[0], e[1], e[2], e[3]
			current_q = q_vals[i]
			if new_state is None:
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

	def train(self, episodes, max_episode_length, sess, env):
		max_reward = 0

		for game in range(episodes):

			total_reward = 0

			env.reset()
			state = env.step(env.action_space.sample())[0]

			for step in range(max_episode_length):
				env.render()
				if np.random.random() < self.epsilon:
					action = env.action_space.sample()
				else:
					action = self.get_action(state, sess).item()
				new_state, reward, done, info = env.step(action)
				total_reward += reward
				self.memory.append((state, action, reward, new_state, done))

				curr_loss = self.update_model(sess)
								
				if done:
					break

				state = new_state
				self.epsilon = max(self.epsilon * self.e_decay, self.e_baseline)

			self.update_target_model(sess)

			if total_reward > max_reward:
					max_reward = total_reward

			if game%50 == 0:
					print("=======================")
					print("Loss:", curr_loss, "Max Reward:", max_reward, "Epsilon", dqn.epsilon)
		    
		env.close()


if __name__ == '__main__':

	env = gym.make('LunarLander-v2')

	num_states = env.env.observation_space.shape[0]
	num_actions = env.env.action_space.n

	hparams = {'n_state_nodes': num_states, 'n_actions': num_actions, 'n_h1': 90, 'n_h2': 30, 'n_h3': 10, 'lr': 0.001, 'discount_rate': 0.85, 'epsilon': 1, 'e_decay': 0.999995, 'e_baseline': 0.1, 'batch_size': 32}

	dqn = DQN(hparams)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		dqn.train(2000, 5000, sess, env)
		
