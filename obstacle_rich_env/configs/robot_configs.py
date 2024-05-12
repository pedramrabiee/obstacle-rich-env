import sys
import inspect

unicycle = dict(
    name='unicycle',
    action_bound=dict(low=(-4.0, -1.0), high=(4.0, 1.0)),
    bound_action=True,
)


double_integrator = dict(
    name='double_integrator',
    action_bound=dict(low=(-4.0, -1.0), high=(4.0, 1.0)),
    bound_action=True,
)


def get_robot_configs(robot_name):
    # Get the current module
    current_module = sys.modules[__name__]

    if hasattr(current_module, robot_name):
        return getattr(current_module, robot_name)
    raise 'robot config is not found'


