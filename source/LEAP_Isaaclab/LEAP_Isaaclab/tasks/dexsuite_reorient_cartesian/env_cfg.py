"""Legacy Dexsuite physics with a separate Cartesian arm action contract."""

from isaaclab.envs.mdp import RelativeJointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from LEAP_Isaaclab.compat.dexsuite.config.kuka_allegro.camera_cfg import (
    StateObservationCfg,
)
from LEAP_Isaaclab.tasks.dexsuite_reorient_pbt.dexsuite_reorient_pbt_env_cfg import (
    DexsuiteReorientPbtEnvCfg,
    DexsuiteReorientPbtEnvCfg_PLAY,
)

from .actions import BoundedPalmPoseActionCfg, palm_controller_state
from .control import FINGER_JOINT_NAMES


@configclass
class CartesianActionsCfg:
    arm: BoundedPalmPoseActionCfg = BoundedPalmPoseActionCfg()
    fingers: RelativeJointPositionActionCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(FINGER_JOINT_NAMES),
        preserve_order=True,
        scale=0.1,
    )


@configclass
class CartesianObservationsCfg(StateObservationCfg):
    @configclass
    class ControllerCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=palm_controller_state)
        enable_corruption = False
        concatenate_terms = True
        history_length = 0

    controller: ControllerCfg = ControllerCfg()


@configclass
class DexsuiteReorientCartesianEnvCfg(DexsuiteReorientPbtEnvCfg):
    actions: CartesianActionsCfg = CartesianActionsCfg()
    observations: CartesianObservationsCfg = CartesianObservationsCfg()


@configclass
class DexsuiteReorientCartesianEnvCfg_PLAY(DexsuiteReorientPbtEnvCfg_PLAY):
    actions: CartesianActionsCfg = CartesianActionsCfg()
    observations: CartesianObservationsCfg = CartesianObservationsCfg()
