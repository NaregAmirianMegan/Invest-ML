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
	def __init__(self, hparams)
		self.hparams = hparams
		self.x, self.y, self.output, self.loss, self.training_op = self._build_model(hparams)

	def _build_model(self, hparams):
		x = tf.placeholder(tf.float32, [None, hparams['num_inputs']])
		y = tf.placeholder(tf.float32, [None, hparams['num_actions']])
		discounted_episode_rewards = tf.placeholder(tf.float32, [None,])

		h1 = tf.layers.dense(x, hparams['n_h1'], activation=tf.nn.relu)
		output = tf.layers.dense(h1, hparams['num_actions'], activation=tf.nn.softmax)

		neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y)
		loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards)
		optimizer = tf.train.AdamOptimizer(learning_rate=hparams['lr'])
		training_op = optimizer.minimize(loss)

		return x, y, output, loss, training_op

	def forward(self, state, sess):
		return sess.run(self.output, feed_dict={self.x: np.reshape(state, (1, self.hparams['num_inputs']))})

	def get_action(self, state, sess):
		probs = self.forward(state, sess)
		highest_prob_action = np.random.choice(self.hparams['num_actions'], p=np.squeeze(probs))
		log_prob = math.log(np.squeeze(probs)[highest_prob_action])
		return highest_prob_action, log_prob

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

	def _normalize(arr):
		arr = (arr - np.mean(arr))/np.std(arr)

	def update_policy(self, rewards, log_probs):
		discounted_rewards = np.array(self._get_discounted_rewards(rewards))
		discounted_rewards = self._normalize(discounted_rewards)

	def train(self, max_episodes, max_episode_len, sess):
		for episode in range(max_episodes):
			for step in range(max_episode_len):
				



	

if __name__ = "__main__":
	env = gym.make('CartPole-v0')

	hparams = {'num_inputs': env.env.observation_space.shape[0], 
			   'num_actions': env.env.action_space.n, 
			   'n_h1': 128, 'lr': 3e-4, 
			   'gamma': 0.9}

	agent = PolicyAgent(hparams)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())



