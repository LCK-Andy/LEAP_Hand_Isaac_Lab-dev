# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""PhysX-only configuration for the original released IsaacGymEnvs DexPBT task."""

from pathlib import Path

from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.utils import configclass

ASSET_DIRECTORY = Path(__file__).resolve().parents[2] / "assets/dexpbt_original"
JOINT_NAMES = tuple(
    [f"iiwa7_joint_{index}" for index in range(1, 8)]
    + [f"{finger}_joint_{index}" for finger in ("index", "middle", "ring", "thumb") for index in range(4)]
)
FINGERTIP_NAMES = ("index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3")
DEFAULT_JOINT_POSITION = (-1.571, 1.571, 0.0, 1.376, 0.0, 1.485, 2.358) + (0.0,) * 16
JOINT_FRICTION = {name: 0.025 for name in JOINT_NAMES[:7]}
JOINT_FRICTION.update(
    {
        f"{finger}_joint_{index}": friction
        for finger in ("index", "middle", "ring")
        for index, friction in enumerate((0.0523963, 0.0150275, 0.00616359, 0.0227036))
    }
)
JOINT_FRICTION.update({f"thumb_joint_{index}": 0.035 for index in range(4)})


def original_robot_cfg(usd_path):
    """Use original touch-sensor URDF dynamics, never the maintained Dexsuite robot cfg."""
    # Spawn's legal thumb pose is only a construction seed; reset deliberately uses original zero-hand baseline.
    spawn_joint_position = dict(zip(JOINT_NAMES, DEFAULT_JOINT_POSITION))
    spawn_joint_position["thumb_joint_0"] = 0.279244444444
    return ArticulationCfg(
        prim_path="/World/envs/env_[^/]+/Robot",
        spawn=UsdFileCfg(
            usd_path=usd_path,
            physics_material=PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.01,
                angular_damping=0.01,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.8, 0.0), joint_pos=spawn_joint_position),
        joint_ordering=JOINT_NAMES,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["iiwa7_joint_[1-7]"],
                stiffness=40.0,
                damping=5.0,
                joint_effort_limit=300.0,
                joint_velocity_limit=10.0,
                armature=0.0,
                friction={name: value for name, value in JOINT_FRICTION.items() if name.startswith("iiwa")},
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["(index|middle|ring|thumb)_joint_[0-3]"],
                stiffness=40.0,
                damping=5.0,
                joint_effort_limit=0.35,
                joint_velocity_limit=6.283,
                armature=0.0,
                friction={name: value for name, value in JOINT_FRICTION.items() if not name.startswith("iiwa")},
            ),
        },
    )


@configclass
class DexPBTOriginalEnvCfg(DirectRLEnvCfg):
    """Original 60Hz control, two PhysX substeps, fixed per-environment cube variants."""

    decimation = 2
    episode_length_s = 600 * 0.01667
    action_space = 23
    observation_space = 117
    state_space = 0
    compute_final_obs = False
    sim = SimulationCfg(
        dt=0.01667 / 2,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physics=PhysxCfg(
            solver_type=1,
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,
        ),
        physics_material=PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
    )
    # Isaac Gym's lower=-spacing and upper=spacing creates a 2*spacing grid.
    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=2.4, replicate_physics=True)
    robot_usd_path = str(ASSET_DIRECTORY / "robot.usd")
    original_profile = "isaacgymenvs-665c321-released"
    original_mutable_params: dict[str, float] = {}
    randomize_object_dimensions = True
    with_small_cuboids = True
    with_big_cuboids = True
    with_sticks = True
    object_base_size = 0.05
    object_density = 400.0
    dof_speed_scale = 10.0
    actions_moving_average = 1.0
    force_scale = 2.0
    force_probability_range = (0.001, 0.1)
    force_decay = 0.99
    force_decay_interval = 0.08
    initial_success_tolerance = 0.075
    target_success_tolerance = 0.01
    tolerance_curriculum_increment = 0.9
    tolerance_curriculum_interval = 3000
    reset_position_noise = (0.1, 0.1, 0.02)
    reset_dof_position_noise_arm = 0.1
    reset_dof_position_noise_fingers = 0.1
    reset_dof_velocity_noise = 0.5


@configclass
class DexPBTOriginalEnvCfg_PLAY(DexPBTOriginalEnvCfg):
    scene = InteractiveSceneCfg(num_envs=16, env_spacing=2.4, replicate_physics=True)
