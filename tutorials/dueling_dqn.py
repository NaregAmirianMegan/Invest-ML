import gym, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#local imports
from dense_nn import _copy_weights, _make_dense_nn_layers

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

class DuelDQN:
	def __init__(self, hparams):
		self.hparams = hparams
		self.lr = hparams['lr']
		self.discount_rate = hparams['discount_rate']

		self.epsilon = hparams['epsilon']
		self.e_decay = hparams['e_decay']
		self.e_baseline = hparams['e_baseline']

		self.memory = CircularBuffer(1000000)

		self.batch_size = hparams['batch_size']

		self.x, self.y, self.output_layer, self.loss, self.training_op = self._create_model(hparams, 'q-model')
		self.x_target, self.y_target, self.output_layer_target, self.loss_target, self.training_op_target = self._create_model(hparams, 'target-model')


	def _create_model(self, hparams, scope):
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			x = tf.placeholder(tf.float32, shape=[None, hparams['n_inputs']])
			y = tf.placeholder(tf.float32, shape=[None, hparams['n_outputs']])

			h1 = tf.layers.dense(x, hparams['hidden_layers']['n_h1'], kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
			h2 = tf.layers.dense(h1, hparams['hidden_layers']['n_h2'], kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu)
			
			value_split, advantage_split = tf.split(h2, 2, axis=1)

			value = tf.layers.dense(value_split, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())
			advantage = tf.layers.dense(advantage_split, hparams['n_outputs'], kernel_initializer=tf.contrib.layers.xavier_initializer())

			normalized_advantage = advantage - tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True)
			output = tf.add(value, normalized_advantage)

			mse = tf.losses.mean_squared_error(y, output)
			loss = tf.reduce_mean(mse)
			optimizer = tf.train.AdamOptimizer(learning_rate=hparams['lr'])
			training_op = optimizer.minimize(loss)

			return x, y, output, loss, training_op

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

	env = gym.make('LunarLander-v2')

	num_states = env.env.observation_space.shape[0]
	num_actions = env.env.action_space.n

	hparams = {'n_inputs': num_states, 
			   'n_outputs': num_actions, 
			   'hidden_layers': {'n_h1': 150, 'n_h2': 120},
			   'lr': 1e-3, 
			   'discount_rate': 0.99, 
			   'epsilon': 1, 
			   'e_decay': 0.995, 
			   'e_baseline': 0.01, 
			   'batch_size': 64}

	dqn = DuelDQN(hparams)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		dqn.train(5000, 5000, sess, env, render_game=False)
		
