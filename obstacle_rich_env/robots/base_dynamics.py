from cbftorch.dynamics import AffineInControlDynamics
import numpy as np
from gymnasium.spaces import Box


class BaseDynamics(AffineInControlDynamics):
    def __init__(self, state_dim, action_dim, params=None, **kwargs):
        super().__init__(state_dim, action_dim, params)
        # Expect action bounds to be a dict of {'low'=(u1_low, u2_low, ...), 'high'=(u1_high, u2_high, ...)}

        if 'bound_action' in self._params and self._params['bound_action']:
            self.action_bound = self._params['action_bound'] if 'action_bound' in self._params else None
        else:
            self.action_bound = None

        self.random_generator = kwargs.get('random_generator', None)

    def initialize_states_from_pos(self, pos):
        """
        Upon initializing placing the robot this method is called to randomize the state are than
         position for robot placement and state initialization.
        """
        raise NotImplementedError

    def zero_pad_states_from_pos(self, pos):
        """
        Given the position of the robot this method is called to initialize other states to zero.
        """
        raise NotImplementedError

    def build_observation_space(self):
        return dict(state=Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float64))

    def build_custom_observation_space(self):
        """
        You can build custom observation space here for your specific purposes. For example,
        you can have cos(theta), sin(theta) in place of theta for your network trainings.
        get_custom_state should be implemented for this purpose.
        """
        return {}

    def get_custom_state(self, state):
        raise NotImplementedError

    def build_action_space(self):
        if self.action_bound is not None:
            return Box(np.array(self.action_bound['low']), np.array(self.action_bound['high']), (self.action_dim,),
                       dtype=np.float64)
        return Box(-np.inf, np.inf, (self.action_dim,), dtype=np.float64)

    def get_robot_pos(self, state):
        raise NotImplementedError

    def get_robot_vel(self, state):
        raise NotImplementedError

    def get_robot_rot(self, state):
        raise NotImplementedError