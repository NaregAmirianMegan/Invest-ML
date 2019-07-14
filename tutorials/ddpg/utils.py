import tensorflow as tf 
import numpy as np 

class ACBuffer:
	def __init__(self, size, input_dims, action_dims):
		self.states = np.zeros((size, input_dims))
		self.actions = np.zeros((size, action_dims))
		self.rewards = np.zeros(size)
		self.new_states = np.zeros((size, input_dims))
		self.terminality = np.zeros(size)
		self.size = size
		self.curr_index = 0
		self.full = False

	def store(self, state, action, reward, new_state, done):
		if self.curr_index == self.size:
			self.full = True
			self.curr_index = 0
		self.states[self.curr_index] = state
		self.actions[self.curr_index] = action
		self.rewards[self.curr_index] = reward 
		self.new_states[self.curr_index] = new_state 
		self.terminality[self.curr_index] = 1 - done
		self.curr_index += 1

	def random_sample(self, batch_size):
		if self.full:
			random_indices = np.random.choice(self.size, batch_size)
			return self.states[random_indices], self.actions[random_indices], self.rewards[random_indices], \
					self.new_states[random_indices], self.terminality[random_indices]
		else:
			random_indices = np.random.choice(self.curr_index, batch_size)
			return self.states[random_indices], self.actions[random_indices], self.rewards[random_indices], \
					self.new_states[random_indices], self.terminality[random_indices]


