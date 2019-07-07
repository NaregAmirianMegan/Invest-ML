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

class PolicyAgent:
	def __init__(self, hparams):
		self.hparams = hparams
		self.memory = CircularBuffer(20000)
		self.build_model(hparams)

	def build_model(self, hparams):
		x = tf.placeholder(tf.float32, [None, hparams['states']])
		y = tf.placeholder(tf.float32, [hparams['actions']])

		h1 = tf.layers.dense(x, hparams['n_h1'], activation=tf.nn.relu)
		h2 = tf.layers.dense(h1, hparams['n_h2'], activation=tf.nn.relu)
		h3 = tf.layers.dense(h2, hparams['n_h3'], activation=tf.nn.relu)
		out = tf.layers.dense(h3, hparams['actions'])

	def get_action(self, state):



	def train(self, episodes, max_episode_length, sess, env):
		max_reward = 0

		for game in range(episodes):

			# total_reward = 0
			total_discounted_reward = 0

			env.reset()
			state = env.step(env.action_space.sample())[0]

			for step in range(max_episode_length):
				env.render()

				action = self.get_action(state)
				
				new_state, reward, done, info = env.step(action)

				log_probs = ???
				self.memory.append((log_probs, reward))

				total_discounted_reward += self.hparams['discount']**step*reward

				# total_reward += reward
								
				if done:
					break

				state = new_state

			# if total_reward > max_reward:
					# max_reward = total_reward

			if game%50 == 0:
					print("=======================")
					print("Max Reward:", max_reward)
		    
		env.close()

if __name__ == '__main__':

	env = gym.make('LunarLander-v2')

	num_states = env.env.observation_space.shape[0]
	num_actions = env.env.action_space.n