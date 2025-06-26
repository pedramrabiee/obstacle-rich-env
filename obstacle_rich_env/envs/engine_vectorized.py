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
from box import Box as AD
from torchdiffeq import odeint
from functools import partial
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
import pygame
import os
from collections import deque
from obstacle_rich_env.envs.utils import FixedSizeTensorStack, vstack_to_tensor

matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plotting

from obstacle_rich_env.envs.engine import Engine, ResamplingError


class EngineVectorized(Engine):
    def __init__(self, config: {}, render_mode=None):
        super().__init__(config, render_mode)

        # set set_num_envs and rebuild observation and action spaces
        # Note that Engine will make spaces for num_envs = 1
        self.set_num_envs(self.config.num_envs)

        self.robot_state_queue = None
        self.goal_pos_queue = None
        self.robot_goal_pairs = []

    def spawn_robot_and_goal(self):
        counter = 0

        # when the environment is not yet reset, then sample as much as self.config.robot_goal_pair_queue_size pair
        # however, if the environment is reset, then sample fewer number of pairs
        if self.robot_state is None:
            self.num_pairs = self.config.robot_goal_pair_queue_size
        else:
            self.num_pairs = max(self.config.robot_goal_pair_queue_size // 10, self.num_envs * 2)

        while len(self.robot_goal_pairs) < self.num_pairs:

            # Sample self.config.robot_goal_pair_queue_size number of valid robot states
            # and store them in the robot_state_queue
            self.spawn_robot()

            # Sample self.config.robot_goal_pair_queue_size number of valid robot states
            # and store them in the robot_state_queue
            self.spawn_goal()

            # Try to pair robots with goals.
            self.pair_robot_with_goal()

            # reset goal_pos_queue
            self.robot_state_queue = None
            self.goal_pos_queue = None

            counter += 1

            if counter > 1000:
                raise ResamplingError('Failed to place enough robot and goal pairs')

    def spawn_robot(self):
        """Sample a new safe robot state"""
        for _ in range(10000):  # Retries
            xy_batch = self.random_generator.uniform(-self.floor_size, self.floor_size,
                                                     (self.config.reset_batch_size, 2))

            robot_state = torch.tensor(self.robot.initialize_states_from_pos(pos=xy_batch), dtype=torch.float64)
            passed_indices = (self.barrier.get_min_barrier_at(robot_state).squeeze(
                dim=1) > self.config.robot_init_thresh).nonzero().squeeze()

            self.robot_state_queue = vstack_to_tensor(self.robot_state_queue, robot_state[passed_indices])

            if self.robot_state_queue is not None and self.robot_state_queue.size(0) >= self.num_pairs:
                self.robot_state_queue = self.robot_state_queue[:self.num_pairs]
                break
        else:
            raise ResamplingError('Failed to place enough robots')

    def spawn_goal(self):
        """Sample a goal position"""
        for _ in range(10000):  # Retries
            xy_batch = self.random_generator.uniform(-self.floor_size, self.floor_size,
                                                     (self.config.reset_batch_size, 2))

            goal_state = self.robot.zero_pad_states_from_pos(torch.tensor(xy_batch))
            passed_indices = (self.barrier.get_min_barrier_at(goal_state).squeeze(
                dim=1) > self.config.goal_init_thresh).nonzero().squeeze()

            self.goal_pos_queue = vstack_to_tensor(self.goal_pos_queue,
                                                   self.robot.get_robot_pos(goal_state[passed_indices]))

            if self.goal_pos_queue is not None and self.goal_pos_queue.size(0) >= self.num_pairs:
                self.goal_pos_queue = self.goal_pos_queue[:self.num_pairs]

                break

        else:
            raise ResamplingError('Failed to place robot')

    def pair_robot_with_goal(self):
        robot_pos_queue = self.robot.get_robot_pos(self.robot_state_queue)

        counter = 0
        while len(robot_pos_queue) > 0:
            random_indices = self.random_generator.permutation(self.goal_pos_queue.size(0))
            self.goal_pos_queue = self.goal_pos_queue[random_indices]

            valid_pairs = torch.norm(self.goal_pos_queue - robot_pos_queue, dim=1) > self.config.min_robot_to_goal_dist

            self.robot_goal_pairs.extend(
                [(self.robot_state_queue[idx.item()], self.goal_pos_queue[idx.item()]) for idx in
                 valid_pairs.nonzero(as_tuple=False)]
            )

            invalid_pairs = ~valid_pairs

            robot_pos_queue = robot_pos_queue[invalid_pairs]
            self.robot_state_queue = self.robot_state_queue[invalid_pairs]
            self.goal_pos_queue = self.goal_pos_queue[invalid_pairs]

            counter += 1
            if counter > 5 or len(self.robot_goal_pairs) >= self.num_pairs:
                break

    def after_spawn(self):
        self._reset_envs()

    def _reset_envs(self, rows_to_reset=None):
        if self.robot_state is None and rows_to_reset is not None:
            raise "Robot state is not initialized"

        # Number of environments to reset
        num_rows_to_reset = self.num_envs if rows_to_reset is None else len(rows_to_reset)
        if num_rows_to_reset == 0:
            return

        robot_states = []
        goal_poses = []
        for _ in range(num_rows_to_reset):
            robot_state, goal_pos = self.robot_goal_pairs.pop()
            robot_states.append(robot_state)
            goal_poses.append(goal_pos)

        # Stack the sampled robot states and goal poses
        robot_states = torch.vstack(robot_states)
        goal_poses = torch.vstack(goal_poses)

        if rows_to_reset is None:
            self.robot_state = robot_states
            self.goal_pos = goal_poses
        else:
            self.robot_state[rows_to_reset] = robot_states
            self.goal_pos[rows_to_reset] = goal_poses

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

        self._ep_step += 1

        # Call render
        if self.render_mode == "human":
            self.render()

        obs = self.obs()

        # Handle resetting environments that are done

        terminated = self.terminated()
        truncated = self.truncated()
        done = (terminated.astype(float) + truncated.astype(float)) >= 1.0
        rows_to_reset = np.where(done)[0]

        # Reset environments that are done
        self._reset_envs(rows_to_reset=rows_to_reset)

        # velocity row reset
        self._vel_queue.reset(rows_to_reset=rows_to_reset)

        # reset self._ep_step to zero
        self._ep_step[rows_to_reset] = 0

        # Resample robot and goal pairs
        if len(self.robot_goal_pairs) < self.num_envs:
            self.spawn_robot_and_goal()

        return obs, reward, terminated, truncated, dict(safety_violated=self._safety_violated(obs),
                                                        success=self._success())

    def set_num_envs(self, num_envs):
        # Set number of environments and rebuild
        # observation and action spaces
        self.num_envs = num_envs
        self.build_observation_space()
        self.build_action_space()
