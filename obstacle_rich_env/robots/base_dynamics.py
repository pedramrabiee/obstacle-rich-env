from hocbf_composition.dynamics import AffineInControlDynamics
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
        Upon initializing placing the robot this method is called to randomize the states are than
         position for robot placement and state initialization.
        """
        raise NotImplementedError

    def build_observation_space(self):
        return dict(states=Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float64))

    def build_action_space(self):
        if self.action_bound is not None:
            return Box(self.action_bound['low'], self.action_bound['high'], (self.action_dim,), dtype=np.float64)
        return Box(-np.inf, np.inf, (self.action_dim,), dtype=np.float64)
