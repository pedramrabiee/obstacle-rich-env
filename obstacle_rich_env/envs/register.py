from gymnasium.envs.registration import register
from obstacle_rich_env.configs.robot_configs import get_robot_configs
from copy import deepcopy
from attrdict import AttrDict as AD
import importlib

ROBOT_NAMES = ['unicycle']
VERSION = 'v0'


class EnvBase:
    ''' Base used to allow for convenient hierarchies of environments '''

    def __init__(self, name='', config={}, prefix='Safe'):
        self.name = name
        self.config = config
        self.robot_configs = {}
        self.prefix = prefix
        for robot_name in ROBOT_NAMES:
            self.robot_configs[robot_name] = {'robot': get_robot_configs(robot_name)}

    def copy(self, name='', config={}):
        new_config = self.config.copy()
        new_config.update(config)
        return EnvBase(self.name + name, new_config)

    def register(self, name='', config={}):
        for robot_name, robot_config in self.robot_configs.items():
            # Default
            env_name = f'{self.prefix}-{robot_name.capitalize()}{self.name.capitalize() + name}-{VERSION}'
            map_config = getattr(importlib.import_module(f'obstacle_rich_env.configs.map_configs'),
                                 f'map_config_{robot_name}')
            config.update({'map_config': map_config})
            reg_config = self.config.copy()
            reg_config.update(robot_config)
            reg_config.update(config)
            register(id=env_name,
                     entry_point='obstacle_rich_env.envs.engine:Engine',
                     kwargs={'config': reg_config})

# =======================================#
# Common Environment Parameter Defaults  #
# =======================================#

base = EnvBase('', getattr(importlib.import_module(f'obstacle_rich_env.configs.common_configs'), 'common_configs'))
zero_base_dict = {}

# =======================================#
#            Goal Environments           #
# =======================================#

goal_all = {'task': 'goal'}

goals = [deepcopy(zero_base_dict), deepcopy(zero_base_dict)]
goal_base = base.copy('Goal', goal_all)

map_layout_module = importlib.import_module(f'obstacle_rich_env.configs.map_layouts')

# Retrieve map config and register environment
for i, goal in enumerate(goals):
    map_layout = getattr(map_layout_module, f'map_layout_goal{i}')
    goal.update({'map_layout': map_layout})
    goal_base.register(str(i), goal)
