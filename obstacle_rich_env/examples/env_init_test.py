import obstacle_rich_env
import gymnasium as gym

env = gym.make('Safe-UnicycleGoal0-v0', render_mode="human")
obs = env.reset(seed=10)
for i in range(1000):
    action = env.action_space.sample()
    obs = env.step(action)
env.close()
