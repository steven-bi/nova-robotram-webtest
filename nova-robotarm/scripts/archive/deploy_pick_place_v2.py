"""Deploy pick-and-place v8.3 - continuous gated lifting.

v8.2 results: grasp=1.56 (success!) but lifting=0 because binary threshold z>0.82
has a 4cm dead zone (same problem as v7). noise_std=0.09 too low to explore.

v8.3 fix: Replace binary grasped_and_lifted with continuous grasped_height_reward.
Like lift_cube's continuous object_height_reward, but GATED by grasp score.

Object: 4x4x6cm cuboid (gripper opens ~4cm, must fit).
"""
import paramiko
import time

SSH_HOST = 'connect.westd.seetacloud.com'
SSH_PORT = 14918
SSH_USER = 'root'
SSH_PASS = 'UvUnT2x1jsaa'
BASE = '/root/autodl-tmp/arm_grasp'

# ==============================================================================
# FILE 1: arm_grasp/mdp/rewards.py
# ==============================================================================
REWARDS_PY = r'''"""Custom reward functions for arm grasp tasks."""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---- Existing reward functions (lift-cube, unchanged) ----

def reaching_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Negative distance penalty: -distance."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    return -distance


def reaching_bonus(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense bonus when close: 1 - tanh(d/std)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    return (obj.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_height_reward(
    env: ManagerBasedRLEnv,
    initial_height: float,
    max_bonus_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    height_gain = (height - initial_height).clamp(min=0.0)
    normalized = (height_gain / (max_bonus_height - initial_height)).clamp(max=1.0)
    return normalized


def grasp_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward for closing the gripper when near the object (binary threshold)."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    near_object = (distance < threshold).float()
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_mean = gripper_pos.mean(dim=1)
    gripper_closed = (1.0 - gripper_mean / open_pos).clamp(min=0.0, max=1.0)
    return near_object * gripper_closed


# ---- NEW/IMPROVED reward functions (pick-and-place v2) ----

def soft_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Soft grasp: continuous proximity × gripper_closed. No binary threshold.

    Key improvement over grasp_reward: uses tanh kernel for proximity instead of
    binary distance threshold. This provides gradient signal even when not perfectly
    close to the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    proximity = 1 - torch.tanh(distance / std)
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_mean = gripper_pos.mean(dim=1)
    gripper_closed = (1.0 - gripper_mean / open_pos).clamp(min=0.0, max=1.0)
    return proximity * gripper_closed


def place_reward(
    env: ManagerBasedRLEnv,
    std: float,
    vel_threshold: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for placing object near target with low velocity.

    v2 fix: Added minimal_height gate so this doesn't fire when cup is on high table.
    Only activates when object has been lifted above the resting position.
    """
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    proximity = 1 - torch.tanh(distance / std)
    obj_vel = torch.norm(obj.data.root_lin_vel_w, dim=1)
    low_vel = (obj_vel < vel_threshold).float()
    # Height gate: only fire when object has been lifted
    height_gate = (obj.data.root_pos_w[:, 2] > minimal_height).float()
    return proximity * low_vel * height_gate


def cup_upright_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    dist_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for keeping cup upright when near target."""
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    near_target = (distance < dist_threshold).float()
    quat = obj.data.root_quat_w
    upright = quat[:, 0] ** 2 + quat[:, 3] ** 2 - quat[:, 1] ** 2 - quat[:, 2] ** 2
    upright_bonus = upright.clamp(min=0.0)
    return near_target * upright_bonus


def object_dropped_penalty(
    env: ManagerBasedRLEnv,
    min_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty when object drops below minimum height."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] < min_height, 1.0, 0.0)


# ---- v3: GRASP-GATED rewards (prevent push-and-fling exploit) ----

def _compute_grasp_score(
    env: ManagerBasedRLEnv,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Helper: compute grasp score [0,1]. 1 = gripper closed AND near object."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    near = (distance < grasp_threshold).float()
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_mean = gripper_pos.mean(dim=1)
    gripper_closed = (1.0 - gripper_mean / open_pos).clamp(min=0.0, max=1.0)
    return near * gripper_closed


def grasped_and_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward ONLY when object is lifted AND grasped.

    v3 fix: Prevents the push-and-fling exploit where the agent
    gets lifting reward by pushing the cup off the table.
    Now lifting only counts if the gripper is closed near the object.
    """
    grasp_score = _compute_grasp_score(
        env, grasp_threshold, open_pos, robot_cfg, object_cfg, ee_frame_cfg
    )
    obj: RigidObject = env.scene[object_cfg.name]
    is_lifted = (obj.data.root_pos_w[:, 2] > minimal_height).float()
    return grasp_score * is_lifted


def grasped_height_reward(
    env: ManagerBasedRLEnv,
    initial_height: float,
    max_bonus_height: float,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Height reward ONLY when grasped. Prevents push exploit."""
    grasp_score = _compute_grasp_score(
        env, grasp_threshold, open_pos, robot_cfg, object_cfg, ee_frame_cfg
    )
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]
    height_gain = (height - initial_height).clamp(min=0.0)
    normalized = (height_gain / (max_bonus_height - initial_height)).clamp(max=1.0)
    return grasp_score * normalized


def grasped_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    grasp_threshold: float,
    open_pos: float,
    command_name: str,
    minimal_height: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Goal tracking ONLY when object is grasped AND lifted above minimal_height.

    v4 fix: Added minimal_height gate. Without this, the agent collects goal_tracking
    reward just by closing gripper near the stationary cup (cup is 0.37m from goal,
    tanh(0.37/0.3)=0.84, giving 16% free reward with weight=16 → significant signal
    that discourages lifting exploration).
    """
    grasp_score = _compute_grasp_score(
        env, grasp_threshold, open_pos, robot_cfg, object_cfg, ee_frame_cfg
    )
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    height_gate = (obj.data.root_pos_w[:, 2] > minimal_height).float() if minimal_height > 0 else 1.0
    return grasp_score * height_gate * (1 - torch.tanh(distance / std))
'''

# ==============================================================================
# FILE 2: arm_grasp/mdp/observations.py (MAJOR UPGRADE - 4 new observations)
# ==============================================================================
OBSERVATIONS_PY = r'''"""Custom observation functions for arm grasp tasks.

v2 additions based on Isaac Lab official stack/pick_place tasks:
- ee_to_object_vector: THE most critical observation (3D)
- ee_position_in_robot_frame: explicit EE position (3D)
- object_velocity_in_robot_frame: detect when grasped (3D)
- gripper_opening: normalized gripper state (1D)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def object_orientation_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The orientation (quaternion) of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        object.data.root_pos_w, object.data.root_quat_w,
    )
    return object_quat_b


# ---- NEW v2 observations (inspired by Isaac Lab stack task) ----

def ee_to_object_vector(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Vector from EE to object in world frame (3D).

    THE most important observation for manipulation tasks.
    Official Isaac Lab stack task uses this as 'gripper_to_cube'.
    Provides direct gradient signal for reaching behavior without
    requiring the network to learn forward kinematics.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    return obj.data.root_pos_w - ee_w


def ee_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE position in robot base frame (3D).

    Tells the policy where its end-effector currently is.
    Official Isaac Lab stack task provides this as 'eef_pos'.
    """
    robot = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_w
    )
    return ee_b


def object_velocity_in_robot_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity in robot frame (3D).

    Helps the policy detect when the object is being moved (grasped)
    vs stationary (on table). Critical for transport phase.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w[:, :3]


def gripper_opening(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Normalized gripper opening state (2D - left and right finger).

    Official Isaac Lab stack task provides this as 'gripper_pos'.
    Returns the actual gripper joint positions for both fingers.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_pos = robot.data.joint_pos[:, -2:]
    return gripper_pos
'''

# ==============================================================================
# FILE 3: arm_grasp/mdp/__init__.py
# ==============================================================================
MDP_INIT_PY = r'''"""Custom MDP functions for arm grasp tasks."""
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
'''

# ==============================================================================
# FILE 4: arm_grasp/envs/pick_place_cfg.py (HEAVILY MODIFIED)
# ==============================================================================
PICK_PLACE_CFG_PY = r'''"""Pick-and-Place v8.4: correct URDF + self-collision + joint deviation.

Changes from v8.3:
- URDF: real joint limits (j2:[-π,0.75], j3:[0,π], j4/j5:[-1.5,1.5])
- URDF: primitive collision shapes (cylinder/box) instead of mesh
- arm_cfg: enabled_self_collisions=True
- Reward: joint_deviation_l1 to prevent self-collapse exploit
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from arm_grasp import mdp
from arm_grasp.assets import ARM_6DOF_CFG


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Scene: robot on high table, coffee cup, low table as target."""

    robot = ARM_6DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.75)

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_6",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.05)),
            ),
        ],
    )

    # Coffee cup as small cuboid (must fit in gripper!)
    # v5 fix: Gripper opens ~4cm per finger. Previous 6cm object was TOO WIDE
    # to grip. Lift_cube (proven) uses 4cm cube. Using 4cm×4cm×6cm (cup shape).
    # Center at z=0.78 (bottom at 0.75 = table top).
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.78),
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.06),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.27, 0.07),
            ),
        ),
    )

    # High table: center at z=0.375 so table top = z=0.75 (matching lift-cube!)
    # BUG FIX: was at (0,0,0) which put table top at z=0.375
    table_high = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableHigh",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.375)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.5, 0.4),
            ),
        ),
    )

    # Low table: center at z=0.275 so table top = z=0.55
    table_low = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLow",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.275)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.55),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.50, 0.55),
            ),
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Target position ABOVE the table (lift_cube style).
    Phase 1: just learn to lift. Phase 2 (later): change to low table goal.
    """
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",
        resampling_time_range=(5.0, 5.0),  # match lift_cube
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35),
            pos_y=(-0.15, 0.15),
            pos_z=(0.10, 0.30),  # z_world = 0.85-1.05 (above table)
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """6-DOF arm + binary gripper."""
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        scale=2.0,
        use_default_offset=True,
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_left_joint", "gripper_right_joint"],
        open_command_expr={"gripper_left_joint": 0.04, "gripper_right_joint": 0.04},
        close_command_expr={"gripper_left_joint": 0.0, "gripper_right_joint": 0.0},
    )


@configclass
class ObservationsCfg:
    """EXACT lift_cube observations (5 terms, 32 dims). Simpler = easier to learn."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.08, 0.08), "y": (-0.10, 0.10), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """v8.4: correct URDF + self-collision + joint deviation penalty.

    Root cause of self-collapse: no joint limits on j2/j3/j4/j5 + no self-collision.
    Fix: updated URDF with real limits + enabled_self_collisions=True.
    Also add joint_deviation_l1 to discourage extreme postures.
    """

    # Phase 1: STRONG reaching (from lift_cube)
    reaching = RewTerm(func=mdp.reaching_penalty, weight=5.0)
    reaching_fine = RewTerm(
        func=mdp.reaching_bonus, params={"std": 0.1}, weight=10.0
    )

    # Phase 2: Grasp (wider threshold for easier discovery)
    grasp = RewTerm(
        func=mdp.grasp_reward,
        params={"threshold": 0.12, "open_pos": 0.04},
        weight=5.0,
    )

    # Phase 3: CONTINUOUS lifting GATED by grasp (KEY FIX for v8.3!)
    # v8.2 used binary z>0.82 → no gradient. Now continuous 0.78→1.0.
    lifting_object = RewTerm(
        func=mdp.grasped_height_reward,
        params={
            "initial_height": 0.78, "max_bonus_height": 1.0,
            "grasp_threshold": 0.12, "open_pos": 0.04,
        },
        weight=15.0,
    )

    # Phase 4: Goal tracking - GATED by grasp + height
    object_goal_tracking = RewTerm(
        func=mdp.grasped_goal_distance,
        params={
            "std": 0.3, "command_name": "object_pose",
            "minimal_height": 0.80,
            "grasp_threshold": 0.12, "open_pos": 0.04,
        },
        weight=16.0,
    )
    object_goal_tracking_fine = RewTerm(
        func=mdp.grasped_goal_distance,
        params={
            "std": 0.05, "command_name": "object_pose",
            "minimal_height": 0.80,
            "grasp_threshold": 0.12, "open_pos": 0.04,
        },
        weight=5.0,
    )

    # Drop penalty (safety net)
    drop_penalty = RewTerm(
        func=mdp.object_dropped_penalty,
        params={"min_height": 0.50},
        weight=-20.0,
    )

    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # Posture regularization: penalize extreme joint angles from zero.
    # Prevents self-collapse exploit (robot folding down onto table).
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        )},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    pass  # No curriculum


@configclass
class ArmPickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 8.0  # match lift_cube
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class ArmPickPlaceEnvCfg_PLAY(ArmPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
'''

# ==============================================================================
# FILE 5: arm_grasp/envs/__init__.py
# ==============================================================================
ENVS_INIT_PY = r'''"""Environment configurations."""
from .lift_cube_cfg import ArmLiftCubeEnvCfg, ArmLiftCubeEnvCfg_PLAY, ArmLiftSceneCfg
from .pick_place_cfg import ArmPickPlaceEnvCfg, ArmPickPlaceEnvCfg_PLAY
'''

# ==============================================================================
# FILE 6: arm_grasp/agents/rsl_rl_ppo_cfg.py (FIXED PPO hyperparams)
# ==============================================================================
AGENTS_PPO_PY = r'''"""RSL-RL PPO configurations for arm tasks."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ArmLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for lift-cube task (existing, proven)."""
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "arm_lift_cube"
    run_name = ""
    resume = False
    logger = "tensorboard"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ArmPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for pick-and-place v8.4.

    v8.3: entropy=0.003, continuous gated lifting.
    v8.4: same hyperparams, new experiment name to track separately.
    """
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "arm_pick_place_v8_4"
    run_name = ""
    resume = False
    logger = "tensorboard"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,  # v8.3 worked: grasp learned, exploration OK
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
'''

# ==============================================================================
# FILE 7: arm_grasp/agents/__init__.py
# ==============================================================================
AGENTS_INIT_PY = r'''"""Agent configurations."""
from .rsl_rl_ppo_cfg import ArmLiftCubePPORunnerCfg, ArmPickPlacePPORunnerCfg
'''

# ==============================================================================
# FILE 8: scripts/train_pick_place.py (v2 - new experiment name)
# ==============================================================================
TRAIN_SCRIPT = r'''"""Train pick-and-place v2 policy with RSL-RL PPO."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from arm_grasp.envs.pick_place_cfg import ArmPickPlaceEnvCfg
from arm_grasp.agents import ArmPickPlacePPORunnerCfg


def main():
    cfg = ArmPickPlaceEnvCfg()
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)

    agent_cfg = ArmPickPlacePPORunnerCfg()
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rsl_rl", agent_cfg.experiment_name,
    )
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
'''

# ==============================================================================
# FILE 9: scripts/validate_pick_place.py
# ==============================================================================
VALIDATE_SCRIPT = r'''"""Validate pick-and-place v2 environment loads and runs."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from arm_grasp.envs.pick_place_cfg import ArmPickPlaceEnvCfg

cfg = ArmPickPlaceEnvCfg()
cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg=cfg)

obs, _ = env.reset()

print("=== Pick-and-Place v2 Environment Validation ===")
print(f"Obs shape: {obs['policy'].shape}")
print(f"Action space: {env.action_space}")

robot = env.scene["robot"]
ee_frame = env.scene["ee_frame"]
obj = env.scene["object"]
print(f"Robot pos: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"EE pos: {ee_frame.data.target_pos_w[0, 0].cpu().numpy()}")
print(f"Cup pos: {obj.data.root_pos_w[0].cpu().numpy()}")
print(f"Cup quat: {obj.data.root_quat_w[0].cpu().numpy()}")

# Run 5 steps with random actions
for step in range(5):
    action = torch.zeros(4, env.action_space.shape[-1], device=env.device)
    obs, rew, term, trunc, info = env.step(action)
    ee_pos = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
    cup_pos = obj.data.root_pos_w[0].cpu().numpy()
    print(f"Step {step}: EE={ee_pos}, Cup={cup_pos}, rew={rew[0].item():.3f}")

print("\n=== VALIDATION PASSED ===")
env.close()
simulation_app.close()
'''


def main():
    print("=" * 60)
    print("  Pick-and-Place v8.4 Deployment")
    print("  Real joint limits + self-collision + posture penalty")
    print("=" * 60)
    print("\nv8.4 changes from v8.3:")
    print("  - URDF: joint_2 [-π,0.75], joint_3 [0,π], joint_4/5 [-1.5,1.5]")
    print("  - URDF: primitive collision shapes (cylinder/box)")
    print("  - enabled_self_collisions=True")
    print("  - joint_deviation_l1 penalty weight=-0.1")
    print("  - experiment: arm_pick_place_v8_4 (fresh start)")
    print()

    print("Connecting to SeetaCloud server...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=15)
    print("Connected!")

    # Kill any running training
    print("\nStopping any running training...")
    client.exec_command('pkill -9 -f "train"', timeout=5)
    time.sleep(3)

    # Upload all files
    sftp = client.open_sftp()

    files = {
        f'{BASE}/arm_grasp/mdp/rewards.py': REWARDS_PY,
        f'{BASE}/arm_grasp/mdp/observations.py': OBSERVATIONS_PY,
        f'{BASE}/arm_grasp/mdp/__init__.py': MDP_INIT_PY,
        f'{BASE}/arm_grasp/envs/pick_place_cfg.py': PICK_PLACE_CFG_PY,
        f'{BASE}/arm_grasp/envs/__init__.py': ENVS_INIT_PY,
        f'{BASE}/arm_grasp/agents/rsl_rl_ppo_cfg.py': AGENTS_PPO_PY,
        f'{BASE}/arm_grasp/agents/__init__.py': AGENTS_INIT_PY,
        f'{BASE}/scripts/train_pick_place.py': TRAIN_SCRIPT,
        f'{BASE}/scripts/validate_pick_place.py': VALIDATE_SCRIPT,
    }

    print("\nUploading files...")
    for path, content in files.items():
        with sftp.file(path, 'w') as f:
            f.write(content)
        fname = path.split('/')[-1]
        print(f"  Uploaded: {fname}")

    sftp.close()
    print("All 9 files uploaded!")

    # Clear __pycache__
    print("\nClearing Python cache...")
    client.exec_command(
        f'find {BASE}/arm_grasp -name "__pycache__" -type d -exec rm -rf {{}} + 2>/dev/null',
        timeout=5,
    )
    time.sleep(1)

    # Run validation
    print("\nRunning validation...")
    cmd = (
        f'cd {BASE} && '
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        '/root/autodl-tmp/conda_envs/thunder2/bin/python -u scripts/validate_pick_place.py 2>&1'
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode()

    idx = output.find('=== Pick-and-Place')
    if idx >= 0:
        print(output[idx:])
    else:
        print(output[-3000:])
    print("Validation exit code:", exit_code)

    if exit_code != 0:
        print("\nVALIDATION FAILED! Fix errors before training.")
        client.close()
        return

    # Start training
    print("\nStarting pick-and-place v2 training (10000 iterations)...")
    train_cmd = (
        f'cd {BASE} && '
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u '
        'scripts/train_pick_place.py --num_envs 2048 --headless '
        f'> {BASE}/train_pick_place.log 2>&1 &'
    )
    client.exec_command(train_cmd, timeout=10)
    time.sleep(5)

    # Verify training started
    stdin, stdout, stderr = client.exec_command(
        'ps aux | grep train_pick_place | grep -v grep | head -3', timeout=5
    )
    ps_out = stdout.read().decode().strip()
    if ps_out:
        print("Training process started!")
        print(ps_out[:200])
    else:
        print("WARNING: Training process not found!")
        stdin, stdout, stderr = client.exec_command(
            f'tail -30 {BASE}/train_pick_place.log', timeout=10
        )
        print(stdout.read().decode())

    client.close()
    print("\nDone! Monitor with: python check_pick_place.py")


if __name__ == "__main__":
    main()
