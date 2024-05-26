from __future__ import annotations
from typing import Any
import gymnasium
import gymnasium.spaces
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box

from obstacle_rich_env.envs.map import Map
from obstacle_rich_env.envs.robot import Robot
import numpy as np
import torch
from attrdict import AttrDict as AD
from torchdiffeq import odeint
from functools import partial
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import pygame
import os
from collections import deque

matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting


class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


class Engine(gymnasium.Env, gymnasium.utils.EzPickle):
    def __init__(self, config: {}, render_mode=None):
        gymnasium.utils.EzPickle.__init__(self, config=config)

        self.config = AD(config)
        self.observation_flatten = self.config['observation_flatten']
        self.timestep = self.config['timestep']
        self.floor_size = np.array(self.config['floor_size'])
        self.render_mode = render_mode

        # Set random seed and make random_generator
        self.set_seed(self.config['seed'] if 'seed' in self.config else None)

        # Instantiate robot
        self.robot = Robot(robot_name=config['robot']['name'], random_generator=self.random_generator)

        # Initialize map to get the barrier dimension for the observation space
        self.map = Map(robot=self.robot, layout=self.config['map_layout'], cfg=self.config,
                       random_generator=self.random_generator)

        # Make spaces
        self.build_observation_space()
        self.action_space = self.robot.build_action_space()

        # Set map back to None
        self.robot_state, self.goal_pos = None, None
        # vel_queue
        self._vel_queue = None

        self.screen = None

    def set_seed(self, seed: int | None = None) -> None:
        """Set internal random next_state seeds."""
        self._seed = 1523876 if seed is None else seed
        self.random_generator = np.random.RandomState(self._seed)

    def build_observation_space(self):
        obs_space_dict = OrderedDict()
        # build observation space
        if 'state' in self.config.obs_key_to_return:
            obs_space_dict.update(self.robot.build_observation_space())

        if 'custom_state' in self.config.obs_key_to_return:
            custom_observation_space = self.robot.build_custom_observation_space()
            assert len(custom_observation_space) > 0, 'custom state is not implemented'
            obs_space_dict.update(custom_observation_space)

        # build goal space
        if 'goal_robot_diff' in self.config.obs_key_to_return:
            obs_space_dict.update(self._build_goal_observation_space())

        # build barrier space
        if 'barriers' in self.config.obs_key_to_return:
            obs_space_dict.update(self._build_barrier_observation_space())

        self.obs_space_dict = gymnasium.spaces.Dict(obs_space_dict)
        self.observation_space = self.obs_space_dict

        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(self.obs_space_dict)
            self.obs_flat_size = self.observation_space.shape[0]

    def reset(self, *,
              seed: int | None = None,
              options: dict[str, Any] | None = None,
              ):
        if seed is not None:
            self.set_seed(seed)

        if self.map is None or self.config.reset_map_layout:
            self.map = Map(robot=self.robot, layout=self.config['map_layout'], cfg=self.config,
                           random_generator=self.random_generator)

        # Spawn robot and sample state
        self.spawn_robot()

        # Spawn goal
        self.spawn_goal()
        self.last_dist_to_goal = self.dist_to_goal()

        self._step = 0

        # Push velocity to memory
        self._vel_queue = None
        self.push_vel()

        # Render
        if self.render_mode == "human":
            self.render()

        obs = self.obs()
        return obs, dict(safety_violated=self._safety_violated(obs), success=self._success())

    def step(self, action):
        action = action if torch.is_tensor(action) else torch.from_numpy(action)
        next_state = odeint(func=lambda t, y: partial(self.robot.dynamics.rhs,
                                                      action=action)(y),
                            y0=self.robot_state,
                            t=torch.tensor([0.0, self.timestep]), method=self.config.integrator)[-1].squeeze().detach()

        self.robot_state = next_state

        # Add velocity to memory
        self.push_vel()

        reward = self.reward()

        self._step += 1

        # Call render
        if self.render_mode == "human":
            self.render()

        obs = self.obs()
        return obs, reward, self.terminated(), self.truncated(), dict(safety_violated=self._safety_violated(obs),
                                                                      success=self._success().item())

    def _safety_violated(self, obs):
        if not self.observation_flatten and 'barriers' in obs:
            return obs['barriers'].min() < 0.0
        else:
            return self.barrier.get_min_barrier_at(self.robot_state).squeeze().item() < 0.0

    def reward(self):
        reward = 0.0
        dist_to_goal = self.dist_to_goal()
        # Distance to goal reward
        reward += (self.last_dist_to_goal.item() - dist_to_goal.item()) * self.config.reward_dist_coef
        self.last_dist_to_goal = dist_to_goal
        # Goal achieved
        if self.goal_met().item():
            reward += self.config.reward_goal_coef
        # Grid lock penalty
        if self.gridlocked():
            reward -= self.config.reward_gridlock_coef
        # Safety violation penalty
        min_barrier = self.barrier.get_min_barrier_at(self.robot_state).squeeze().item()
        if min_barrier < 0:
            reward += min_barrier * self.config.reward_safety_coef

        return reward

    def obs(self):
        obs = {}
        if 'state' in self.config.obs_key_to_return:
            obs.update({'state': self.robot_state_np})
        if 'custom_state' in self.config.obs_key_to_return:
            obs.update({'custom_state': self.robot.get_custom_state(self.robot_state).squeeze().cpu().detach().numpy()})
        if 'goal_robot_diff' in self.config.obs_key_to_return:
            obs.update({'goal_robot_diff': self.goal_pos_np - self.robot.get_robot_pos(self.robot_state).squeeze().cpu().detach().numpy()})
        if 'barriers' in self.config.obs_key_to_return:
            obs.update({'barriers': torch.hstack(
                self.barrier.compute_barriers_at(self.robot_state)).squeeze().cpu().detach().numpy()})

        if self.observation_flatten:
            obs = gymnasium.spaces.utils.flatten(self.obs_space_dict, obs)

        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'

        return obs

    def terminated(self):
        return self._step > self.config.max_episode_steps

    def truncated(self):
        return (self.barrier.get_min_barrier_at(
            self.robot_state) < self.config.barrier_truncation_thresh).squeeze().item()

    def _success(self):
        return self.dist_to_goal() < self.config.goal_size

    def spawn_robot(self):
        """Sample a new safe robot state"""
        for _ in range(10000):  # Retries
            xy_batch = self.random_generator.uniform(-self.floor_size, self.floor_size,
                                                     (self.config.reset_batch_size, 2))
            robot_state = torch.tensor(self.robot.initialize_states_from_pos(pos=xy_batch), dtype=torch.float64)
            passed_indices = (self.barrier.get_min_barrier_at(robot_state).squeeze(
                dim=1) > self.config.robot_init_thresh).nonzero().squeeze()
            if passed_indices.numel() > 0:
                index = passed_indices[0]
                self.robot_state = robot_state[index]
                break
        else:
            raise ResamplingError('Failed to place robot')

    def spawn_goal(self):
        """Sample a goal position"""
        for _ in range(10000):  # Retries
            xy_batch = self.random_generator.uniform(-self.floor_size, self.floor_size,
                                                     (self.config.reset_batch_size, 2))
            xy_batch = torch.tensor(xy_batch, dtype=torch.float64)
            suitable_indices = self.dist_to_robot(xy_batch).squeeze() > self.config.min_robot_to_goal_dist
            xy_batch = xy_batch[suitable_indices]

            if len(xy_batch) > 0:
                goal_state = self.robot.zero_pad_states_from_pos(xy_batch)
                passed_indices = (self.barrier.get_min_barrier_at(goal_state).squeeze(
                    dim=1) > self.config.goal_init_thresh).nonzero().squeeze()
                if passed_indices.numel() > 0:
                    index = passed_indices[0]
                    # The robot.get_robot_pos method extract the position date from the state data and returns it
                    self.goal_pos = self.robot.get_robot_pos(goal_state[index])
                    break
        else:
            raise ResamplingError('Failed to place robot')

    def dist_to_robot(self, xy: torch.tensor):
        if xy.ndim == 2:
            return torch.norm(self.robot.get_robot_pos(self.robot_state) - xy, keepdim=True, dim=1)
        return torch.norm(self.robot.get_robot_pos(self.robot_state) - xy)

    def dist_to_goal(self):
        return self.dist_to_robot(self.goal_pos)

    def goal_met(self):
        return self.dist_to_goal() < self.config.goal_size

    def gridlocked(self):
        if not self.goal_met():
            vels = torch.stack(tuple(self._vel_queue))
            rms = torch.norm(vels) / torch.sqrt(torch.tensor(vels.numel()))
            if rms.item() < self.config.gridlock_threshold:
                return True
        return False

    def push_vel(self):
        if self._vel_queue is None:
            self._vel_queue = deque([torch.tensor(self.config.gridlock_threshold * self.config.gridlock_check_duration)] * self.config.gridlock_check_duration)
        self._vel_queue.appendleft(self.robot.get_robot_vel(self.robot_state))
        while len(self._vel_queue) > self.config.gridlock_check_duration:
            self._vel_queue.pop()

    def close(self):  # Part of Gym interface
        if self.screen is not None:
            os.remove(self.map_image)
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    @property
    def robot_state_np(self):
        return self.robot_state.squeeze().cpu().detach().numpy()

    @property
    def goal_pos_np(self):
        return self.goal_pos.cpu().detach().numpy()

    @property
    def barrier(self):
        return self.map.barrier

    def render(self):
        if self.screen is None:
            # Initialize Pygame
            pygame.init()

            # Set up the display
            self.width, self.height = 800, 800
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Obstacle Rich Environment")
            else:
                self.screen = pygame.Surface((self.width, self.height))

            # Generate map image using Matplotlib
            self.map_image = self._generate_map_image()

            # Load the map image into Pygame
            self.map_surface = pygame.image.load(self.map_image)
            self.map_surface = pygame.transform.scale(self.map_surface, (self.width, self.height))

            # Initialize robot and goal sprites
            self.robot_sprite = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(self.robot_sprite, (135, 206, 250, 128), (20, 20), 20)  # Robot circle
            pygame.draw.circle(self.robot_sprite, (255, 255, 255), (20, 20), 5)  # Robot center dot

            self.goal_sprite = pygame.Surface((40, 40), pygame.SRCALPHA)
            pygame.draw.circle(self.goal_sprite, (144, 238, 144, 128), (20, 20), 20)  # Goal circle
            pygame.draw.circle(self.goal_sprite, (255, 255, 255), (20, 20), 5)  # Goal center dot

            # Set robot and goal positions
            self.robot_pos_render = self._coords_to_pixel(self.robot_state_np[:2])
            self.goal_pos_render = self._coords_to_pixel(self.goal_pos_np)

        else:
            # Clear the screen
            self.screen.fill((255, 255, 255))

            # Draw the map image
            self.screen.blit(self.map_surface, (0, 0))

            # Update robot position
            self.robot_pos_render = self._coords_to_pixel(self.robot_state_np[:2])

            # Draw the robot
            robot_rect = self.robot_sprite.get_rect(center=self.robot_pos_render)
            self.screen.blit(self.robot_sprite, robot_rect)

            # Draw the goal
            goal_rect = self.goal_sprite.get_rect(center=self.goal_pos_render)
            self.screen.blit(self.goal_sprite, goal_rect)

        if self.render_mode == "human":
            # Update the display
            pygame.display.flip()
        else:
            # Get the pixel data from the current screen
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def _generate_map_image(self):
        X, Y, Z = self._generate_map_contours()

        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        cmap = matplotlib.colors.ListedColormap(['#FFF5F5', '#FFD8D8'])  # Soft color palette
        ax.contourf(X, Y, Z, levels=[-10, 0], colors=cmap.colors)
        ax.contour(X, Y, Z, levels=[0], colors='#FF6961')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-self.floor_size[0], self.floor_size[0])
        ax.set_ylim(-self.floor_size[1], self.floor_size[1])

        ax.axis('off')

        image_path = 'map_image.png'
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=600)
        plt.close(fig)

        return image_path

    def _build_goal_observation_space(self):
        return dict(goal_robot_diff=Box(-np.inf, np.inf, (2,), dtype=np.float64))

    def _build_barrier_observation_space(self):
        return dict(barriers=Box(-np.inf, np.inf, (self.barrier.num_barriers,), dtype=np.float64))

    def _generate_map_contours(self):
        x = np.linspace(-self.floor_size[0], self.floor_size[0], 500)
        y = np.linspace(-self.floor_size[1], self.floor_size[1], 500)
        X, Y = np.meshgrid(x, y, )
        points = np.column_stack((X.flatten(), Y.flatten()))
        points = torch.tensor(points, dtype=torch.float64)
        points = self.robot.zero_pad_states_from_pos(points)
        Z = self.map.barrier.min_barrier(points)
        Z = Z.reshape(X.shape)
        return X, Y, Z

    def _coords_to_pixel(self, coords):
        x, y = coords
        pixel_x = int((x + self.floor_size[0]) * (self.width / (2 * self.floor_size[0])))
        pixel_y = int((-y + self.floor_size[1]) * (self.height / (2 * self.floor_size[1])))
        return pixel_x, pixel_y

    def make_obs_functional(self, obs_keys):
        obs_funcs = {
            'state': lambda x: x,
            'custom_state': lambda x: self.robot.get_custom_state(x),
            'goal_robot_diff': lambda x: self.goal_pos.to(x.device).repeat(x.shape[0], 1) - self.robot.get_robot_pos(x),
            'barriers': lambda x: torch.hstack(self.barrier.compute_barriers_at(x)),
        }
        req_funcs = [obs_funcs[key] for key in obs_keys]
        return lambda x: torch.cat([func(x) for func in req_funcs], dim=-1).detach()

    def set_config(self, new_config: dict):
        for k, v in new_config.items():
            assert k in self.config, f"Key {k} is not a valid config key."
            self.config[k] = v
