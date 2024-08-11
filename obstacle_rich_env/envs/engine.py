from __future__ import annotations
from typing import Any
import gymnasium
import gymnasium.spaces
from gymnasium.core import RenderFrame
from gymnasium.spaces import Box
from obstacle_rich_env.envs.map import Map, EmptyMap
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
from obstacle_rich_env.envs.utils import FixedSizeTensorStack
from obstacle_rich_env.envs.lidar import ObstacleLidar

matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting


class ResamplingError(AssertionError):
    ''' Raised when we fail to sample a valid distribution of objects or goals '''
    pass


class Engine(gymnasium.Env, gymnasium.utils.EzPickle):
    def __init__(self, config: {}, render_mode=None):
        gymnasium.utils.EzPickle.__init__(self, config=config)
        self.num_envs = 1

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
        self.initialize_map()

        # Make spaces
        self.build_observation_space()
        self.build_action_space()

        # Set map back to None
        self.robot_state, self.goal_pos = None, None

        # Set velocity queue to None
        self._vel_queue = None

        # Set rendering screen to None
        self.screen = None

        if 'obstacle_lidar' in self.config.obs_key_to_return or 'obstacle_lidar_coor' in self.config.obs_key_to_return:
            self.obstacle_lidar = ObstacleLidar(self.map, self.robot, self.config)
            self.robot.mount_obstacle_lidar(self.obstacle_lidar)


        # make trajectory time
        self._traj_times = torch.linspace(0.0, self.timestep, self.config.action_repeat + 1)

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

        if 'min_barrier' in self.config.obs_key_to_return:
            obs_space_dict.update(self._build_min_barrier_observation_space())

        if 'obstacle_lidar' in self.config.obs_key_to_return:
            obs_space_dict.update(self._build_lidar_observation_space())

        if 'obstacle_lidar_coor' in self.config.obs_key_to_return:
            obs_space_dict.update(self._build_lidar_coor_observation_space())

        self.obs_space_dict = gymnasium.spaces.Dict(self._vectorize_obs_space(obs_space_dict))
        self.observation_space = self.obs_space_dict

        if self.observation_flatten:
            self.observation_space = gymnasium.spaces.utils.flatten_space(self.obs_space_dict)
            self.obs_flat_size = self.observation_space.shape[0]

    def build_action_space(self):
        self.action_space = self._vectorize_obs_space(self.robot.build_action_space())
        self.action_space.seed(self._seed)

    def initialize_map(self):
        if self.config['map_is_off'] or len(self.config['map_layout']) == 0:
            self.map = EmptyMap(robot=self.robot, layout=self.config['map_layout'], cfg=self.config,
                                random_generator=self.random_generator)
        else:
            self.map = Map(robot=self.robot, layout=self.config['map_layout'], cfg=self.config,
                           random_generator=self.random_generator)

    def reset(self, *,
              seed: int | None = None,
              options: dict[str, Any] | None = None,
              ):
        if seed is not None:
            self.set_seed(seed)

        if self.map is None or self.config.reset_map_layout:
            self.initialize_map()

        # Spawn robot and goal
        self.spawn_robot_and_goal()

        # Do after spawn tasks
        self.after_spawn()

        # Store last distance to goal for reward calculation
        self.last_dist_to_goal = self.dist_to_goal()

        # Reset step counter
        self._ep_step = np.array([0] * self.num_envs)

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
        if action.ndim == 1:
            action = action.unsqueeze(0)
        next_state = odeint(func=lambda t, y: partial(self.robot.dynamics.rhs,
                                                      action=action)(y),
                            y0=self.robot_state,
                            t=self._traj_times, method=self.config.integrator)[-1].detach()

        self.robot_state = next_state

        # Add velocity to memory
        self.push_vel()

        reward = self.reward()

        self._ep_step += 1

        # Call render
        if self.render_mode == "human":
            self.render()

        obs = self.obs()
        return obs, reward, self.terminated(), self.truncated(), dict(safety_violated=self._safety_violated(obs),
                                                                      success=self._success())

    def _safety_violated(self, obs):
        if not self.observation_flatten and 'barriers' in obs:
            res = (obs['barriers'].min(axis=-1, keepdims=True) < 0.0).astype(float)
        else:
            res = (self.barrier.get_min_barrier_at(self.robot_state).squeeze(0).cpu().numpy() < 0.0).astype(float)

        return res.squeeze().item() if self.num_envs == 1 else res

    def reward(self):
        reward = torch.zeros((self.robot_state.shape[0], 1))
        dist_to_goal = self.dist_to_goal()

        # Distance to goal reward
        reward += (self.last_dist_to_goal - dist_to_goal) * self.config.reward_dist_coef
        self.last_dist_to_goal = dist_to_goal

        # Goal achieved
        condition = self.dist_to_goal() - self.config.goal_size
        reward += torch.where(condition < 0.0, self.config.reward_goal_coef * (1.0 - condition / self.config.goal_size),
                              0.0)

        # Grid lock penalty
        reward -= self.gridlocked() * self.config.reward_gridlock_coef

        # Safety violation penalty
        min_barrier = self.barrier.get_min_barrier_at(self.robot_state) - self.config.barrier_reward_activation_thresh
        reward += torch.where(min_barrier < 0.0,
                              min_barrier * self.config.reward_safety_coef,
                              0.0)

        reward = reward if len(reward) > 1 else reward.squeeze()
        return reward.cpu().detach().numpy()

    def obs(self):
        obs = {}
        if 'state' in self.config.obs_key_to_return:
            obs.update({'state': self.robot_state_np})

        if 'custom_state' in self.config.obs_key_to_return:
            obs.update(
                {'custom_state': self.robot.get_custom_state(self.robot_state).squeeze(0).cpu().detach().numpy()})

        if 'goal_robot_diff' in self.config.obs_key_to_return:
            obs.update({'goal_robot_diff': self.goal_pos_np - self.robot.get_robot_pos(
                self.robot_state).squeeze(0).cpu().detach().numpy()})

        if 'barriers' in self.config.obs_key_to_return:
            obs.update({'barriers': torch.hstack(
                self.barrier.compute_barriers_at(self.robot_state)).squeeze(0).cpu().detach().numpy()})

        if 'min_barrier' in self.config.obs_key_to_return:
            obs.update(
                {'min_barrier': self.barrier.get_min_barrier_at(self.robot_state).squeeze(0).cpu().detach().numpy()})

        if 'obstacle_lidar' in self.config.obs_key_to_return and 'obstacle_lidar_coor' in self.config.obs_key_to_return:
            dist, coord = self.robot.obstacle_lidar.get_lidar_and_coor(self.robot_state)
            obs.update({
                'obstacle_lidar': dist.squeeze(0).cpu().detach().numpy(),
                'obstacle_lidar_coor': coord.squeeze(0).cpu().detach().numpy()
            })
        else:
            if 'obstacle_lidar' in self.config.obs_key_to_return:
                obs.update({'obstacle_lidar': self.robot.obstacle_lidar.get_lidar(self.robot_state).squeeze(
                    0).cpu().detach().numpy()})

            if 'obstacle_lidar_coor' in self.config.obs_key_to_return:
                obs.update({'obstacle_lidar_coor': self.robot.obstacle_lidar.get_lidar_coor(self.robot_state).squeeze(
                    0).cpu().detach().numpy()})

        if self.observation_flatten:
            if self.num_envs > 1:
                flattened_obs = []
                for key in self.obs_space_dict.spaces.keys():
                    flattened_obs.append(obs[key].reshape(obs[key].shape[0], -1))
                obs = np.concatenate(flattened_obs, axis=1)
            else:
                obs = gymnasium.spaces.utils.flatten(self.obs_space_dict, obs)

        assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'

        return obs

    def terminated(self):
        _terminated = self._ep_step > self.config.max_episode_steps
        return _terminated[0] if self.num_envs == 1 else np.expand_dims(_terminated, 1)

    def truncated(self):
        _truncated = (self.barrier.get_min_barrier_at(
            self.robot_state) < self.config.barrier_truncation_thresh).squeeze(0).cpu().detach().numpy()

        return _truncated[0] if self.num_envs == 1 else _truncated

    def _success(self):
        res = (self.dist_to_goal() < self.config.goal_size).double().numpy()
        return res.squeeze().item() if self.num_envs == 1 else res

    def spawn_robot_and_goal(self):
        # Spawn robot and sample state
        self.spawn_robot()

        # Spawn goal
        self.spawn_goal()

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
                self.robot_state = robot_state[index].unsqueeze(0)
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
                    self.goal_pos = self.robot.get_robot_pos(goal_state[index]).unsqueeze(0)
                    break
        else:
            raise ResamplingError('Failed to place robot')

    def after_spawn(self):
        pass

    def dist_to_robot(self, xy: torch.tensor):
        if xy.ndim == 2:
            return torch.norm(self.robot.get_robot_pos(self.robot_state) - xy, keepdim=True, dim=1)
        return torch.norm(self.robot.get_robot_pos(self.robot_state) - xy)

    def dist_to_goal(self):
        return self.dist_to_robot(self.goal_pos)

    def goal_met(self):
        return (self.dist_to_goal() < self.config.goal_size).double()

    def gridlocked(self):
        goal_not_met = 1.0 - self.goal_met()
        vel_cond = (torch.sqrt(
            torch.mean(self._vel_queue.stack ** 2, dim=-1, keepdim=True)) < self.config.gridlock_threshold).double()
        return goal_not_met * vel_cond

    def push_vel(self):
        if self._vel_queue is None:
            self._vel_queue = FixedSizeTensorStack(batch_size=self.num_envs,
                                                   max_length=self.config.gridlock_check_duration,
                                                   init_value=self.config.gridlock_threshold * self.config.gridlock_check_duration)

        self._vel_queue.append(self.robot.get_robot_vel(self.robot_state).unsqueeze(1))

    def close(self):  # Part of Gym interface
        if self.screen is not None:
            os.remove(self.map_image)
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    @property
    def robot_state_np(self):
        return self.robot_state.squeeze(0).cpu().detach().numpy()

    @property
    def goal_pos_np(self):
        return self.goal_pos.squeeze(0).cpu().detach().numpy()

    @property
    def barrier(self):
        return self.map.barrier

    def render(self):
        if self.screen is None:
            # Initialize Pygame
            pygame.init()

            # Set up the display
            self._make_screen_size()
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

            self._robot_sprite_radius = 20
            self._goal_sprite_radius = int(self.config.goal_size / (2 * self.floor_size[0]) * self.width)

            # Initialize robot and goal sprites
            self._create_sprites()
            self._update_render_position()
        else:
            # Clear the screen
            self.screen.fill((255, 255, 255))

            # Draw the map image
            self.screen.blit(self.map_surface, (0, 0))

            # Update render positions and draw
            self._update_render_position()

        if self.render_mode == "human":
            # Update the display
            pygame.display.flip()
        else:
            # Get the pixel data from the current screen
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def _make_screen_size(self):
        max_width = 1920
        max_height = 1080
        aspect_ratio = self.floor_size[0] / self.floor_size[1]

        if max_height * aspect_ratio <= max_width:
            self.height = max_height
            self.width = int(max_height * aspect_ratio)
        else:
            self.width = max_width
            self.height = int(max_width / aspect_ratio)

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

    def _create_sprites(self):
        self.robot_sprites = []
        self.goal_sprites = []
        colors = self._generate_colors(self.num_envs)
        for i in range(self.num_envs):
            self.robot_sprites.append(self._create_circle_sprite(circle_color=colors[i],
                                                                 dot_color=(255, 255, 255),
                                                                 radius=self._robot_sprite_radius))

            self.goal_sprites.append(self._create_circle_sprite(circle_color=colors[i],
                                                                dot_color=(0, 0, 0),
                                                                radius=self._goal_sprite_radius,
                                                                draw_cross=True))

    def _update_render_position(self):
        # Set robot and goal positions
        if self.num_envs == 1:
            robot_pos_render, goal_pos_render = self._update_robot_goal_pos(self.robot_state_np, self.goal_pos_np)
            self._draw_robot_goal(self.robot_sprites[0], robot_pos_render, self.goal_sprites[0], goal_pos_render)
        else:
            for i in range(self.num_envs):
                robot_pos_render, goal_pos_render = self._update_robot_goal_pos(self.robot_state_np[i],
                                                                                self.goal_pos_np[i])
                self._draw_robot_goal(self.robot_sprites[i], robot_pos_render, self.goal_sprites[i], goal_pos_render)

    def _draw_robot_goal(self, robot_sprite, robot_pos_render, goal_sprite, goal_pos_render):
        self.screen.blit(robot_sprite, robot_sprite.get_rect(center=robot_pos_render))
        self.screen.blit(goal_sprite, goal_sprite.get_rect(center=goal_pos_render))

    def _update_robot_goal_pos(self, robot_state, goal_pos):
        robot_pos_render = self._coords_to_pixel(
            self.robot.get_robot_pos(robot_state))
        goal_pos_render = self._coords_to_pixel(goal_pos)
        return robot_pos_render, goal_pos_render

    def _create_circle_sprite(self, circle_color, dot_color, radius=20, draw_cross=False):
        diameter = 2 * radius
        sprite = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
        pygame.draw.circle(sprite, circle_color, (radius, radius), radius)

        if draw_cross:
            cross_size = int(radius * 0.6)
            pygame.draw.line(sprite, dot_color, (radius - cross_size, radius), (radius + cross_size, radius), 2)
            pygame.draw.line(sprite, dot_color, (radius, radius - cross_size), (radius, radius + cross_size), 2)
        else:
            pygame.draw.circle(sprite, dot_color, (radius, radius), int(radius * 0.25))  # Draw dot

        return sprite

    def _generate_colors(self, num_colors):
        np.random.seed(0)  # For reproducibility
        colors = np.random.randint(0, 255, (num_colors, 3))
        colors = [(r, g, b, 128) for r, g, b in colors]
        return colors

    def _build_goal_observation_space(self):
        return dict(goal_robot_diff=Box(-np.inf, np.inf, (2,), dtype=np.float64))

    def _build_barrier_observation_space(self):
        return dict(barriers=Box(-np.inf, np.inf, (self.barrier.num_barriers,), dtype=np.float64))

    def _build_min_barrier_observation_space(self):
        return dict(min_barrier=Box(-np.inf, np.inf, (1,), dtype=np.float64))

    def _build_lidar_observation_space(self):
        return dict(obstacle_lidar=Box(-np.inf, np.inf, (self.config.ray_num,), dtype=np.float64))

    def _build_lidar_coor_observation_space(self):
        if self.config.return_cartesian:
            return dict(obstacle_lidar_coor=Box(-np.inf, np.inf, (self.config.ray_num, 2), dtype=np.float64))
        else:
            # TODO: Fix the bounds based on max_range and scan_angle
            return dict(obstacle_lidar_coor=Box(-np.inf, np.inf, (self.config.ray_num, 2), dtype=np.float64))

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

    def _vectorize_obs_space(self, obs_space):
        if self.num_envs == 1:
            return obs_space

        def proccess_obs_space(space):
            new_shape = (self.num_envs,) + space.shape
            new_low = np.broadcast_to(space.low, new_shape)
            new_high = np.broadcast_to(space.high, new_shape)
            return gymnasium.spaces.Box(low=new_low, high=new_high, shape=new_shape,
                                        dtype=space.dtype)

        if isinstance(obs_space, gymnasium.spaces.Box):
            return proccess_obs_space(obs_space)

        elif isinstance(obs_space, dict) or isinstance(obs_space, gymnasium.spaces.Dict):
            vectorized_obs_space = OrderedDict()
            for key, space in obs_space.items():
                if isinstance(space, gymnasium.spaces.Box):
                    vectorized_obs_space[key] = proccess_obs_space(space)
                else:
                    raise NotImplementedError(f"Vectorization for space type {type(space)} is not implemented.")
        else:
            raise NotImplementedError(f"Vectorization for space type {type(space)} is not implemented.")
        return vectorized_obs_space if isinstance(obs_space, dict) else gymnasium.spaces.Dict(vectorized_obs_space)

    def make_obs_functional(self, obs_keys):
        obs_funcs = {
            'state': lambda x: x,
            'custom_state': lambda x: self.robot.get_custom_state(x),
            'goal_robot_diff': lambda x: self.goal_pos.to(x.device).repeat(x.shape[0] // self.num_envs,
                                                                           1) - self.robot.get_robot_pos(x),
            'barriers': lambda x: torch.hstack(self.barrier.compute_barriers_at(x)),
            'min_barriers': lambda x: self.barrier.get_min_barrier_at(x),
            'obstacle_lidar': lambda x: self.robot.lidar.get_lidar(x),
            'obstacle_lidar_coor': lambda x: self.robot.lidar.get_lidar_coor(x)
        }
        req_funcs = [obs_funcs[key] for key in obs_keys]
        return lambda x: torch.cat([func(x) for func in req_funcs], dim=-1).detach()

    def set_config(self, new_config: dict):
        for k, v in new_config.items():
            assert k in self.config, f"Key {k} is not a valid config key."
            self.config[k] = v
