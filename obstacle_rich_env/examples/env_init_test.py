import obstacle_rich_env
# import gymnasium as gym
from time import time

start_time = time()
env = obstacle_rich_env.make('Unicycle-Goal3', render_mode="human")
obs, _ = env.reset(seed=10)
print(obs)
for i in range(100):
    action = env.action_space.sample()
    obs, _, _, _, _ = env.step(action)
    print(f'Observation at step {i}: {obs}\n')
env.close()
print(time() - start_time)

