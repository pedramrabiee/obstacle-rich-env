import importlib.util
from obstacle_rich_env.configs.robot_configs import get_robot_configs
from obstacle_rich_env.envs.lidar import Lidar



class Robot:
    def __init__(self, robot_name, random_generator):
        self.robot_name = robot_name
        self._params = get_robot_configs(robot_name)
        self.random_generator = random_generator
        self.dynamics = self.make_dynamics()

    def make_dynamics(self):
        module = importlib.import_module(f'obstacle_rich_env.robots.{self.robot_name}')
        class_name = f'{self.robot_name.capitalize()}Dynamics'
        if hasattr(module, class_name):
            dynamics_class = getattr(module, class_name)
            return dynamics_class(params=self._params, random_generator=self.random_generator)
        raise 'Robot dynamics not defined'

    def mount_lidar(self, map, config):
        self.lidar = Lidar(map, self, config)

    def initialize_states_from_pos(self, pos):
        return self.dynamics.initialize_states_from_pos(pos=pos)

    def zero_pad_states_from_pos(self, pos):
        return self.dynamics.zero_pad_states_from_pos(pos=pos)

    def build_observation_space(self):
        return self.dynamics.build_observation_space()

    def build_custom_observation_space(self):
        return self.dynamics.build_custom_observation_space()

    def build_action_space(self):
        return self.dynamics.build_action_space()

    def get_custom_state(self, state):
        return self.dynamics.get_custom_state(state)

    def get_robot_pos(self, state):
        return self.dynamics.get_robot_pos(state)

    def get_robot_vel(self, state):
        return self.dynamics.get_robot_vel(state)

    def get_robot_rot(self, state):
        return self.dynamics.get_robot_rot(state)

    @property
    def state_dim(self):
        return self.dynamics.state_dim

    @property
    def action_dim(self):
        return self.dynamics.action_dim
