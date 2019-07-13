import gym, random, math
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

class PolicyAgent:
	def __init__(self, hparams):
		self.hparams = hparams
		self.x, self.y, self.discounted_rewards, self.output, self.loss, self.training_op, self.sampled_actions = self._build_model(hparams)

	def _build_model(self, hparams):
		x = tf.placeholder(tf.float32, [None, hparams['num_inputs']])
		y = tf.placeholder(tf.float32, [None, hparams['num_actions']])
		discounted_rewards = tf.placeholder(tf.float32, [None,])

		h1 = tf.layers.dense(x, hparams['n_h1'], activation=tf.nn.tanh)
		#h2 = tf.layers.dense(h1, hparams['n_h2'], activation=tf.nn.relu)
		output = tf.layers.dense(h1, hparams['num_actions'])

		sampled_actions = tf.squeeze(tf.multinomial(output, 1))

		neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)
		loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
		optimizer = tf.train.AdamOptimizer(learning_rate=hparams['lr'])
		training_op = optimizer.minimize(loss)

		return x, y, discounted_rewards, output, loss, training_op, sampled_actions

	def _get_discounted_rewards(self, rewards):
		d_rewards = []
		for i in range(len(rewards)):
			d_future_r = 0
			pwr = 0
			for j in rewards[i:]:
				d_future_r += (self.hparams['gamma']**pwr) * j
				pwr += 1
			d_rewards.append(d_future_r)
		return d_rewards

	def _normalize(self, arr):
		mean = np.mean(arr)
		stdev = np.std(arr)
		return (arr - mean)/stdev

	def _discount_and_normalize(self, rewards):
		rewards = np.array(self._get_discounted_rewards(rewards))
		return self._normalize(rewards)

	def predict(self, state, sess):
		return sess.run(self.output, feed_dict={self.x: np.reshape(state, (1, self.hparams['num_inputs']))})

	def get_action(self, state, sess):
		return sess.run(self.sampled_actions, {self.x: np.reshape(state, (1, self.hparams['num_inputs']))})
	
	def train_batch(self, x_batch, y_batch, discounted_rewards, sess):
		_, loss = sess.run([self.training_op, self.loss], feed_dict={self.x: x_batch, self.y: y_batch, self.discounted_rewards: discounted_rewards})
		return loss

	def train(self, max_episodes, max_episode_len, sess, render_game=False):
		reward_sequence = []

		for episode in range(max_episodes):
			episode_state_list = []
			episode_rewards = []
			episode_action_dists = []

			env.reset()
			state = env.step(env.action_space.sample())[0]

			for step in range(max_episode_len):
				if render_game:
					env.render()

				action = self.get_action(state, sess)

				new_state, reward, done, info = env.step(action)

				action_dist = np.zeros((self.hparams['num_actions']))
				action_dist[action] = 1

				episode_state_list.append(state)
				episode_rewards.append(reward)
				episode_action_dists.append(action_dist)

				if done:
					reward_sequence.append(sum(episode_rewards))
					normal_discounted_rewards = self._discount_and_normalize(episode_rewards)
					x_batch = np.vstack(np.array(episode_state_list))
					y_batch = np.vstack(np.array(episode_action_dists))

					loss = self.train_batch(x_batch, y_batch, normal_discounted_rewards, sess)

					if episode % 100 == 0:
						print("GAME", episode, "Avg. R:", sum(reward_sequence[-100:])/100)

					if episode == 3000:
						print('made it')
						render_game = True

					break

				state = new_state

	

if __name__ == "__main__":
	env = gym.make('LunarLander-v2')

	hparams = {'num_inputs': env.observation_space.shape[0], 
			   'num_actions': env.action_space.n, 
			   'n_h1': 40, 
			   'n_h2': 20,
			   'lr': 1e-2, 
			   'gamma': 0.99}

	agent = PolicyAgent(hparams)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		agent.train(5000, 5000, sess)


