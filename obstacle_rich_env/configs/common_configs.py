from attrdict import AttrDict as AD

common_configs = AD(
    timestep=0.01,
    max_episode_steps=1000,
    observation_flatten=True,
    floor_size=[10.0, 10.0],        # half floor size in x and y direction
    reset_map_layout=True,
    obstacle_size_range=[1.0, 4.0],
    robot_init_thresh=0.1,  # barrier function value should be greater than this value when robot is placed
    goal_init_thresh=0.1,
    goal_size=0.3,
    reset_batch_size=1000,  # the number of samples to use in each iteration when safe spawning robot and goal
    min_robot_to_goal_dist=4.0,
    integrator='euler',  # options: euler, rk4, dopri5
    barrier_truncation_thresh=-0.1,  # if the value of barrier goes below this, the episode is truncated
)
