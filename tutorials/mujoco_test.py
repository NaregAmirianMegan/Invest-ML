import gym
env = gym.make('FetchPush-v1')
env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample()
  state, reward, done, info = env.step(action)

  if _ % 100 == 0:
  	print(action)