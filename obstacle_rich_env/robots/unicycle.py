import torch
from obstacle_rich_env.robots.base_dynamics import BaseDynamics
import numpy as np
from gymnasium.spaces import Box


class UnicycleDynamics(BaseDynamics):
    def __init__(self, state_dim=4, action_dim=2, params=None, **kwargs):
        super().__init__(state_dim, action_dim, params, **kwargs)

    def _f(self, x):
        return torch.stack([x[:, 2] * torch.cos(x[:, 3]),
                            x[:, 2] * torch.sin(x[:, 3]),
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return (torch.vstack([torch.zeros(2, 2, dtype=torch.float64),
                              torch.eye(2, dtype=torch.float64)])
                ).repeat(x.shape[0], 1, 1).to(x.device)

    def build_custom_observation_space(self):
        # q_x, q_y, v
        box1 = Box(-np.inf, np.inf, (self.state_dim - 1,), dtype=np.float64)
        # cos(theta), sin(theta)
        box2 = Box(-1.0, 1.0, (2,), dtype=np.float64)
        return dict(custom_state=Box(
            np.concatenate((box1.low, box2.low)),
            np.concatenate((box1.high, box2.high)),
            dtype=np.float64
        ))

    def get_custom_state(self, state):
        if state.ndim == 1:
            return torch.hstack(
                (state[:3], torch.cos(state[3]), torch.sin(state[3])))
        if state.ndim == 2:
            return torch.hstack(
                (state[..., :3], torch.cos(state[..., 3]).unsqueeze(1), torch.sin(state[..., 3]).unsqueeze(1)))

    def initialize_states_from_pos(self, pos):
        return np.concatenate([pos, np.zeros((pos.shape[0], 1)),
                               self.random_generator.uniform(low=0, high=2 * np.pi, size=(pos.shape[0], 1))], axis=1)

    def zero_pad_states_from_pos(self, pos):
        if torch.is_tensor(pos):
            res = torch.zeros((pos.shape[0], self._state_dim))
            res[:, :2] = pos
            return res

        res = np.zeros((pos.shape[0], self._state_dim))
        res[:, :2] = pos
        return res

    def get_robot_pos(self, state):
        if state.ndim == 1:
            return state[:2]
        return state[:, :2]

    def get_robot_vel(self, state):
        if state.ndim == 1:
            return state[2]
        return state[:, 2]

    def get_robot_rot(self, state):
        if state.ndim == 1:
            return state[3]
        return state[:, 3]