from attrdict import AttrDict as AD

common_configs = AD(
    seed=125392,
    timestep=0.05,
    max_episode_steps=10,
    observation_flatten=False,
    obs_key_to_return=["state", "custom_state", "goal_robot_diff", "barriers"],        # provide a list of keys to include in the observation
    floor_size=[10.0, 10.0],        # half floor size in x and y direction
    reset_map_layout=True,
    obstacle_size_range=[1.0, 4.0],
    robot_init_thresh=0.1,  # barrier function value should be greater than this value when robot is placed
    goal_init_thresh=0.1,
    goal_size=1.0,
    reset_batch_size=1000,  # the number of samples to use in each iteration when safe spawning robot and goal
    min_robot_to_goal_dist=0.2,
    integrator='euler',  # options: euler, rk4, dopri5
    barrier_truncation_thresh=-0.05,  # if the value of barrier goes below this, the episode is truncated
    barrier_reward_activation_thresh=0.05,
    tensor_return=True,
    reward_dist_coef=1.0,
    reward_goal_coef=1.0,
    reward_safety_coef=10.0,
    reward_gridlock_coef=1.0,
    gridlock_check_duration=10, #number of timesteps to keep the velocity data for gridlock check
    gridlock_threshold=0.01,
    # vectorized env configs
    num_envs=1,
    robot_goal_pair_queue_size=1000,
)
