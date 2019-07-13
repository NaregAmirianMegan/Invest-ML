'''
state -> actor -> highest value action
state, action -> critic -> score 
'''
import tensorflow as tf 
import numpy as np 
import gym

from sklearn import preprocessing as skp

class Critic:
	def __init__(self, hparams, env):
		self.env = env
		self.hparams = hparams
		self.value_output, self.x = self._build_model(hparams)

	def _build_model(self, hparams):
		var_init = tf.contrib.layers.xavier_initializer()

		x = tf.placeholder(tf.float32, [None, hparams['n_inputs']])

		h1 = tf.layers.dense(x, hparams['hidden_layers']['n_h1'], tf.nn.elu, var_init)
		h2 = tf.layers.dense(h1, hparams['hidden_layers']['n_h2'], tf.nn.elu, var_init)

		output = tf.layers.dense(h2, 2, None, var_init) 

		return output, x

class Actor:
	def __init__(self, hparams, env):
		self.env = env
		self.hparams = hparams
		self.select_action, self.normal_dist, self.x = self._build_model(hparams)

	def _build_model(self, hparams):
		var_init = tf.contrib.layers.xavier_initializer()

		x = tf.placeholder(tf.float32, [None, hparams['n_inputs']])

		h1 = tf.layers.dense(x, hparams['hidden_layers']['n_h1'], tf.nn.elu, var_init)
		h2 = tf.layers.dense(h1, hparams['hidden_layers']['n_h2'], tf.nn.elu, var_init)

		sigma = tf.layers.dense(h2, 1, None, var_init)
		mu = tf.layers.dense(h2, 1, None, var_init)

		sigma = tf.nn.softplus(sigma) + 1e-5
		normal_dist = tf.contrib.distributions.Normal(mu, sigma)
		select_action = tf.squeeze(normal_dist.sample(1), axis=0)
		select_action = tf.clip_by_value(select_action, self.env.action_space.low[0], self.env.action_space.high[0])

		return select_action, normal_dist, x

class ActorCritic:
	def __init__(self, actor_hparams, critic_hparams, env):
		self.env = env
		self.actor = Actor(actor_hparams, env)
		self.critic = Critic(critic_hparams, env)

		self.actor_train_op, self.actor_loss, self.critic_train_op, self.critic_loss, self.action, self.delta, self.target= self._build_actor_critic_model()

		self._input_scaler = self._create_input_scalar(10000)

	def _create_input_scalar(self, num_samples):
		state_samples = np.array([self.env.observation_space.sample() for i in range(num_samples)])
		return skp.StandardScaler().fit(state_samples)

	def _scale_input_data(self, state):
		return self._input_scaler.transform([state])

	def _build_actor_critic_model(self):
		# chosen action at timestep
		action = tf.placeholder(tf.float32)
		# squared temporal difference error
		delta = tf.placeholder(tf.float32)
		# reward plus future discounted reward
		target = tf.placeholder(tf.float32)

		# calculate log gradient for actor loss (+ 1e-5 for log(0) problem)
		actor_loss = -tf.log(self.actor.normal_dist.prob(action)+1e-5)*delta
		actor_train_op = tf.train.AdamOptimizer(learning_rate=self.actor.hparams['lr']).minimize(actor_loss)

		critic_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.critic.value_output), target))
		critic_train_op = tf.train.AdamOptimizer(learning_rate=self.critic.hparams['lr']).minimize(critic_loss)

		return actor_train_op, actor_loss, critic_train_op, critic_loss, action, delta, target

	def train(self, train_params, sess, render=False):
		episode_history = []
		for episode in range(train_params['episodes']):
			state = env.reset()
			episode_reward = 0
			for step in range(train_params['max_episode_len']):
				if render: env.render()
				# sample action
				action = sess.run(self.actor.select_action, feed_dict={self.actor.x: self._scale_input_data(state)})

				new_state, reward, done, info = env.step(np.squeeze(action, axis=0))

				episode_reward += reward

				value_new_state = sess.run(self.critic.value_output, feed_dict={self.critic.x: self._scale_input_data(new_state)})

				target = reward + train_params['gamma']*np.squeeze(value_new_state)

				td_error = target - np.squeeze(sess.run(self.critic.value_output, feed_dict={self.critic.x: self._scale_input_data(state)}))

				# train actor
				_, actor_loss = sess.run([self.actor_train_op, self.actor_loss], feed_dict={self.action: np.squeeze(action), 
																							self.actor.x: self._scale_input_data(state),
																							self.delta: td_error})

				# train critic
				_, critic_loss = sess.run([self.critic_train_op, self.critic_loss], feed_dict={self.critic.x: self._scale_input_data(state),
																							   self.target: target})
				if done: 
					break
				state = new_state

			episode_history.append(episode_reward)

			print("Episode: {}, Cumulative reward: {}".format(episode, episode_reward))
        
			if np.mean(episode_history[-100:]) > 90 and len(episode_history) >= 101:
				print("****************Solved***************")
				print("Mean cumulative reward over last 100 episodes:", np.mean(episode_history[-100:]))


if __name__ == '__main__':
	env = gym.make('MountainCarContinuous-v0')

	num_states = env.observation_space.shape[0]

	critic_hparams = {
			    'n_inputs': num_states,
			    'hidden_layers': {'n_h1': 400, 'n_h2': 400},
			    'lr': 5.6e-4
			  }
	actor_hparams = {
			    'n_inputs': num_states,
			    'hidden_layers': {'n_h1': 40, 'n_h2': 40},
			    'lr': 1e-5
			  }

	train_params = {
				'episodes': 250,
				'max_episode_len': 5000,
				'gamma': 0.99
			  }

	A2C = ActorCritic(actor_hparams, critic_hparams, env)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		A2C.train(train_params, sess, render=True)


