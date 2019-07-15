import tensorflow as tf
import numpy as np 

import gym

from utils import ACBuffer

class Actor:
	def __init__(self, lr, n_actions, n_h1, n_h2, n_inputs, batch_size, action_bound, sess, name="Actor"):
		self.lr = lr
		self.n_actions = n_actions
		self.n_h1 = n_h1
		self.n_h2 = n_h2
		self.n_inputs = n_inputs
		self.batch_size = batch_size
		self.action_bound = action_bound
		self.sess = sess
		self._build_model(name)
		self.network_params = tf.trainable_variables(scope=name)

		self.raw_actor_grads = tf.gradients(self.output, self.network_params, -self.action_gradient)
		self.actor_grads = list(map(lambda x: tf.div(x, self.batch_size), self.raw_actor_grads))

		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(zip(self.actor_grads, self.network_params))


	def _build_model(self, scope):
		w_init = tf.contrib.layers.xavier_initializer()

		with tf.variable_scope(scope):
			self.x = tf.placeholder(tf.float32, [None, self.n_inputs])
			self.action_gradient = tf.placeholder(tf.float32, [None, self.n_actions])


			h1 = tf.layers.dense(self.x, units=self.n_h1, kernel_initializer=w_init, bias_initializer=w_init)
			h1 = tf.nn.relu(tf.layers.batch_normalization(h1))

			h2 = tf.layers.dense(h1, units=self.n_h2, kernel_initializer=w_init, bias_initializer=w_init)
			h2 = tf.nn.relu(tf.layers.batch_normalization(h2))

			output = tf.layers.dense(h2, units=self.n_actions, activation='tanh',
									 kernel_initializer=w_init, bias_initializer=w_init)

			self.output = tf.multiply(output, self.action_bound)

	def predict(self, state):
		return self.sess.run(self.output, feed_dict={self.x: state})

	def train(self, states, gradients):
		return self.sess.run(self.train_op, feed_dict={self.x: states, self.action_gradient: gradients})


class Critic:
	def __init__(self, lr ,n_actions, n_h1, n_h2, n_inputs, batch_size, sess, name="Critic"):
		self.lr = lr
		self.n_actions = n_actions
		self.n_h1 = n_h1
		self.n_h2 = n_h2
		self.n_inputs = n_inputs
		self.batch_size = batch_size
		self.sess = sess
		self._build_model(name)
		self.network_params = tf.trainable_variables(scope=name)

		self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		self.action_gradients = tf.gradients(self.value, self.actions)


	def _build_model(self, scope):
		w_init = tf.contrib.layers.xavier_initializer()

		with tf.variable_scope(scope):
			self.x = tf.placeholder(tf.float32, [None, self.n_inputs])
			self.actions = tf.placeholder(tf.float32, [None, self.n_actions])
			self.q_target = tf.placeholder(tf.float32, [None, 1])


			h1 = tf.layers.dense(self.x, units=self.n_h1, kernel_initializer=w_init, bias_initializer=w_init)
			h1 = tf.nn.relu(tf.layers.batch_normalization(h1))

			h2 = tf.layers.dense(h1, units=self.n_h2, kernel_initializer=w_init, bias_initializer=w_init)
			h2 = tf.layers.batch_normalization(h2)

			action_input = tf.layers.dense(self.actions, units=self.n_h2, activation=tf.nn.relu)

			state_actions = tf.add(h2, action_input)
			state_actions = tf.nn.relu(state_actions)

			self.value = tf.layers.dense(state_actions, units=1, kernel_initializer=w_init, 
										 bias_initializer=w_init, kernel_regularizer=tf.keras.regularizers.l2(0.01))

			self.loss = tf.losses.mean_squared_error(self.q_target, self.value)

	def predict(self, state, actions):
		return self.sess.run(self.value, feed_dict={self.x: state, self.actions: actions})

	def train(self, states, actions, q_target):
		return self.sess.run(self.train_op, feed_dict={self.x: states, self.actions: actions, self.q_target: q_target})

	def get_action_grads(self, states, actions):
		return self.sess.run(self.action_gradients, feed_dict={self.x: states, self.actions: actions})

class ActorCritic:
	def __init__(self, hparams, sess, env):
		self.n_inputs = hparams['n_inputs']
		self.tau = hparams['tau']
		self.gamma = hparams['gamma']
		self.n_actions = hparams['n_actions']
		self.memory = ACBuffer(hparams['buffer_size'], hparams['n_inputs'], hparams['n_actions'])
		self.batch_size = hparams['batch_size']
		self.sess = sess

		self.actor = Actor(lr=hparams['alpha'], n_actions=hparams['n_actions'], n_h1=hparams['n_h1'], n_h2=hparams['n_h2'], 
						   n_inputs=hparams['n_inputs'], batch_size=hparams['batch_size'], action_bound=hparams['action_bound'],
						   sess=sess)

		self.critic = Critic(lr=hparams['beta'], n_actions=hparams['n_actions'], n_h1=hparams['n_h1'], n_h2=hparams['n_h2'], 
						     n_inputs=hparams['n_inputs'], batch_size=hparams['batch_size'], sess=sess)

		self.target_actor = Actor(lr=hparams['alpha'], n_actions=hparams['n_actions'], n_h1=hparams['n_h1'], n_h2=hparams['n_h2'], 
								  n_inputs=hparams['n_inputs'], batch_size=hparams['batch_size'], action_bound=hparams['action_bound'],
								  sess=sess, name="Target_Actor")

		self.target_critic = Critic(lr=hparams['beta'], n_actions=hparams['n_actions'], n_h1=hparams['n_h1'], n_h2=hparams['n_h2'], 
						   			n_inputs=hparams['n_inputs'], batch_size=hparams['batch_size'], sess=sess, name="Target_Critic")

		self.update_actor = [self.target_actor.network_params[i].assign(
											tf.multiply(self.actor.network_params[i], self.tau) + 
											tf.multiply(self.actor.network_params[i], 1. - self.tau))
							  for i in range(len(self.target_actor.network_params))]


		self.update_critic = [self.target_critic.network_params[i].assign(
											tf.multiply(self.critic.network_params[i], self.tau) + 
											tf.multiply(self.critic.network_params[i], 1. - self.tau))
							  for i in range(len(self.target_critic.network_params))]

		sess.run(tf.global_variables_initializer())

		self.update_network_params(first=True)

	def update_network_params(self, first=False):
		if first:
			old_tau = self.tau 
			self.tau = 1.0
			self.target_critic.sess.run(self.update_critic)
			self.target_actor.sess.run(self.update_actor)
			self.tau = old_tau
		else:
			self.target_critic.sess.run(self.update_critic)
			self.target_actor.sess.run(self.update_actor)

	def record(self, state, action, reward, new_state, done):
		self.memory.store(state, action, reward, new_state, done)

	def get_action(self, state):
		state = state[np.newaxis, :]
		action = self.actor.predict(state)
		return action[0]

	def learn(self):
		if self.memory.curr_index <= self.batch_size and self.memory.full == False:
			return

		states, actions, rewards, new_states, terminality = self.memory.random_sample(self.batch_size)

		target_critic_values = self.target_critic.predict(new_states, self.target_actor.predict(new_states))

		q_targets = []
		for i in range(self.batch_size):
			q_targets.append(rewards[i] + self.gamma*target_critic_values[i]*terminality[i])
		q_targets = np.reshape(q_targets, (self.batch_size, 1))

		self.critic.train(states, actions, q_targets)

		action_outs = self.actor.predict(states)
		grads = self.critic.get_action_grads(states, action_outs)
		self.actor.train(states, grads[0])

		self.update_network_params()

if __name__ == '__main__':

	env = gym.make('HandManipulatePen-v0')

	hparams = {
			   'n_inputs': env.observation_space['observation'].shape[0]+env.observation_space['desired_goal'].shape[0]+env.observation_space['achieved_goal'].shape[0],
			   'n_actions': len(env.action_space.high),
			   'buffer_size': 1000000,
			   'batch_size': 64,
			   'action_bound': env.action_space.high,
			   'alpha': 0.00005,
			   'beta': 0.0005,
			   'gamma': 0.99,
			   'tau': 0.001,
			   'n_h1': 400,
			   'n_h2': 200,
			  }
	with tf.Session() as sess:
		agent = ActorCritic(hparams, sess, env)

		score_history = []

		for episode in range(5000):
			state = env.reset()
			state = np.concatenate((state['observation'], state['achieved_goal'], state['desired_goal']))
			done = False
			episode_reward = 0
			while not done:
				action = agent.get_action(state)
				new_state, reward, done, info = env.step(action)

				new_state = np.concatenate((new_state['observation'], new_state['achieved_goal'], new_state['desired_goal']))

				agent.record(state, action, reward, new_state, int(done))
				agent.learn()
				episode_reward += reward 
				state = new_state
				env.render()

			score_history.append(episode_reward)
			print('Episode:', episode, 'Score:', episode_reward, 'Avg. R.', np.mean(score_history[-100:]))

























