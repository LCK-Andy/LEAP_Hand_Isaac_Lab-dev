# --------------------------------------------------------
# LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning
# https://arxiv.org/abs/2309.06440
# Copyright (c) 2025 Kenneth Shaw, Sri Anumakonda
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# v1 env: the official shared InHandManipulationEnv extended with a
# downward-facing RGBD camera observation ("image" group). The 124-d "policy"
# observation, actions, rewards, and resets are inherited unchanged.
# --------------------------------------------------------

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import Camera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import InHandManipulationEnv

from .leap_direct_v1_env_cfg import LeapHandDirectV1EnvCfg


class LeapHandDirectV1Env(InHandManipulationEnv):
    """InHandManipulationEnv + downward-facing RGBD camera observation (v1).

    The official direct in-hand environment (``Isaac-Repose-Cube-LEAP-Direct-v0``)
    with an added ``"image"`` observation group from a world-fixed camera above
    the LEAP palm: rgb (3) + clipped/normalized depth (1), shape (N, 4, H, W).
    The ``"policy"`` group stays the official 124-d "full" vector.
    """

    cfg: LeapHandDirectV1EnvCfg

    def __init__(self, cfg: LeapHandDirectV1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # expose the camera image as a top-level observation group
        c, h, w = self.cfg.rgbd_channels, self.cfg.camera.height, self.cfg.camera.width
        self.single_observation_space["image"] = gym.spaces.Box(0.0, 1.0, (c, h, w), dtype=np.float32)
        self.observation_space = gym.vector.utils.batch_space(
            gym.spaces.Dict(self.single_observation_space.spaces), self.num_envs
        )

    def _setup_scene(self):
        # same as InHandManipulationEnv._setup_scene, plus the RGBD camera
        # (created before cloning so every env gets an instance)
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object: Articulation | RigidObject = self.cfg.object_cfg.class_type(self.cfg.object_cfg)
        self._joint_wrench_sensor = None
        if self.cfg.asymmetric_obs:
            self._joint_wrench_sensor = self._create_joint_wrench_sensor()
        # add the downward-facing RGBD camera
        self.camera = Camera(self.cfg.camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["camera"] = self.camera
        if self._joint_wrench_sensor is not None:
            self.scene.sensors["joint_wrench"] = self._joint_wrench_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_observations(self) -> dict:
        observations = super()._get_observations()

        # append the camera image as a separate observation group
        rgb = self.camera.data.output["rgb"].float() / 255.0
        depth = torch.clamp(self.camera.data.output["depth"], min=0.0, max=self.cfg.rgbd_depth_clip)
        depth = depth / self.cfg.rgbd_depth_clip
        depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
        rgbd = torch.cat([rgb, depth], dim=-1)  # depth output is already (N, H, W, 1)
        rgbd = rgbd.permute(0, 3, 1, 2)  # (N, H, W, 4) -> (N, 4, H, W) for a CNN
        observations["image"] = rgbd

        return observations
