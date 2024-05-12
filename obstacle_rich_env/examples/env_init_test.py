import obstacle_rich_env
# import gymnasium as gym

env = obstacle_rich_env.make('Unicycle-Goal0', render_mode="human")
obs = env.reset(seed=10)
print(obs)
for i in range(1000):
    action = env.action_space.sample()
    obs = env.step(action)
env.close()
