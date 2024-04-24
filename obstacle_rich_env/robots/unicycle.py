import torch
from obstacle_rich_env.robots.base_dynamics import BaseDynamics
import numpy as np


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
                ).repeat(x.shape[0], 1, 1)

    def initialize_states_from_pos(self, pos):
        return np.concatenate([pos, np.zeros((pos.shape[0], 1)),
                               self.random_generator.uniform(low=0, high=2 * np.pi, size=(pos.shape[0], 1))], axis=1)
