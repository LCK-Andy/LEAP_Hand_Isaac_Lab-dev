# --------------------------------------------------------
# LEAP Hand: YCB / varied object reorientation config.
#
# Inherits from leap_hand_env_cfg.LeapHandEnvCfg and overrides
# only the object to a non-DexCube model.
#
# Setup:
#   1. python scripts/download_ycb.py       # download OBJ meshes
#   2. python scripts/convert_ycb_to_usd.py  # create USD wrappers
#   3. python scripts/rl_games/train.py --task Isaac-Reorient-YCB-Leap --num_envs 1024
# --------------------------------------------------------

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from LEAP_Isaaclab.tasks.leap_hand_reorient.leap_hand_env_cfg import LeapHandEnvCfg

# Package assets directory for locally-bundled objects
_HERE = Path(__file__).parent
_YCB_DIR = str(_HERE.parent.parent / "assets" / "ycb")

# -- Object catalog -----------------------------------------------------------

_OBJECTS: dict[str, str] = {
    # YCB objects — bundled locally (OBJ + USD wrapper)
    "cracker_box": f"{_YCB_DIR}/cracker_box.usd",
    "sugar_box": f"{_YCB_DIR}/sugar_box.usd",
    "tomato_soup_can": f"{_YCB_DIR}/tomato_soup_can.usd",
    "mustard_bottle": f"{_YCB_DIR}/mustard_bottle.usd",
    "tuna_fish_can": f"{_YCB_DIR}/tuna_fish_can.usd",
    "pudding_box": f"{_YCB_DIR}/pudding_box.usd",
    "gelatin_box": f"{_YCB_DIR}/gelatin_box.usd",
    "potted_meat_can": f"{_YCB_DIR}/potted_meat_can.usd",
    "master_chef_can": f"{_YCB_DIR}/master_chef_can.usd",
    # YCB fruits
    "banana": f"{_YCB_DIR}/banana.usd",
    "strawberry": f"{_YCB_DIR}/strawberry.usd",
    "apple": f"{_YCB_DIR}/apple.usd",
    "lemon": f"{_YCB_DIR}/lemon.usd",
    "peach": f"{_YCB_DIR}/peach.usd",
    "pear": f"{_YCB_DIR}/pear.usd",
    "orange": f"{_YCB_DIR}/orange.usd",
    # Other manipulable objects
    "wood_block": f"{_YCB_DIR}/wood_block.usd",
    "foam_brick": f"{_YCB_DIR}/foam_brick.usd",
    "rubiks_cube": f"{_YCB_DIR}/rubiks_cube.usd",
    # Nucleus colored blocks (fallback, always available)
    "red_block": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
    "blue_block": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
    "green_block": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/green_block.usd",
    "yellow_block": f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/yellow_block.usd",
}


def _build_object_cfg(
    object_name: str = "cracker_box",
    density: float = 400.0,
    spawn_z: float = 0.56,
) -> RigidObjectCfg:
    """Build a RigidObjectCfg for a catalog object.

    Args:
        object_name: Key into ``_OBJECTS``.
        density: Material density in kg/m³.
        spawn_z: Z position of the object at spawn. Higher = further above palm.
    """
    usd_path = _OBJECTS[object_name]

    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=density),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.00, -0.1, spawn_z),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


# -- Config class --------------------------------------------------------------

_DEFAULT_OBJECT = "banana"
_DEFAULT_DENSITY = 400.0
_DEFAULT_GOAL_SCALE = (1.2, 1.2, 1.2)


@configclass
class LeapHandReorientYCBCfg(LeapHandEnvCfg):
    """Reorientation task config for YCB and other varied objects.

    Available objects (set ``object_name``):
        YCB (19): cracker_box, sugar_box, tomato_soup_can, mustard_bottle,
                  tuna_fish_can, pudding_box, gelatin_box, potted_meat_can,
                  master_chef_can, banana, strawberry, apple, lemon, peach,
                  pear, orange, wood_block, foam_brick, rubiks_cube
        Nucleus: red_block, blue_block, green_block, yellow_block
    """

    object_name: str = _DEFAULT_OBJECT
    object_density: float = _DEFAULT_DENSITY

    # Wider hand opening for larger YCB objects.
    # Values are set just within each joint's position limits to
    # maximize aperture while avoiding self-clipping into the hand.
    # Joint limits from PhysX simulation parameters:
    #   a_0/a_4/a_7: [-1.047, 1.047]  (abduction — keep centered)
    #   a_1/a_8:     [-0.314, 2.230]  (MCP flexion)
    #   a_2/a_6/a_10:[-0.506, 1.885]  (PIP flexion)
    #   a_3/a_11:    [-0.366, 2.042]  (DIP flexion)
    #   a_5/a_13:    [-0.470, 2.443]  (index MCP/DIP)
    #   a_9/a_14:    [-1.200, 1.900]  (ring abduction/flexion)
    #   a_12:        [-0.349, 2.094]  (middle MCP)
    #   a_15:        [-1.340, 1.880]  (little abduction)
    default_joint_pos: list = [
        [0.000, 2.000, 0.000, 0.000,   # a_0..a_3  (thumb)
         -1.000, 2.200, 0.000, 1.000,  # a_4..a_7  (index)
         2.000, 1.800, 1.800, 2.000,   # a_8..a_11 (middle)
         0.000, 2.200, 0.000, 0.000],  # a_12..a_15 (ring + little)
    ]

    # Spawn object higher in the palm to clear fully-opened fingers.
    # DexCube: z=0.56; YCB objects: z=0.62 (6cm above default).
    _OBJECT_SPAWN_Z: float = 0.62

    def __post_init__(self):
        usd_path = _OBJECTS[self.object_name]

        self.object_cfg = _build_object_cfg(
            object_name=self.object_name,
            density=self.object_density,
            spawn_z=self._OBJECT_SPAWN_Z,
        )

        self.goal_object_cfg.markers["goal"] = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=_DEFAULT_GOAL_SCALE,
        )

        if hasattr(self.sim, "physics"):
            self.sim.physics.gpu_max_rigid_contact_count = 2**23
            self.sim.physics.gpu_max_rigid_patch_count = 2**23

        super().__post_init__()