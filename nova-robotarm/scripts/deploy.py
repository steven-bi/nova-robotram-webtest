"""Deploy pick-and-place v8.10 - fix two confirmed bugs from v8.9.

Bug 1: drop_penalty never fires (threshold 0.50 < termination 0.60)
  - Isaac Lab computes reward THEN checks termination each step
  - Object drops to z=0.61 -> reward computed: 0.61 > 0.50 -> penalty=0
  - Then termination fires: 0.61 < 0.60 -> episode ends
  - Result: robot has ZERO explicit penalty for knocking object off table
  Fix: drop_penalty min_height 0.50 -> 0.72 (fires 3cm below table surface)
       object_dropping termination  0.60 -> 0.65 (fires 10cm below table surface)
       Now penalty fires BEFORE termination, creating a gradient signal.

Bug 2: grasp_threshold=0.12m too large
  - Object is only 4x4cm wide. At 12cm distance, gripper fingers (max 4cm opening)
    physically cannot wrap around the object.
  - grasped_object_upvel fires even when EE is 12cm away = no real grasp
  - Reduces specificity of all grasp-gated rewards
  Fix: grasp_threshold 0.12 -> 0.09m (still generous, but physically meaningful)
       At 9cm distance with correct approach angle, gripper can reach object sides.

No other changes from v8.9 (gripper starts OPEN, velocity reward structure kept).
"""
import paramiko
import time

SSH_HOST = 'connect.westd.seetacloud.com'
SSH_PORT = 14918
SSH_USER = 'root'
SSH_PASS = 'UvUnT2x1jsaa'
BASE = '/root/autodl-tmp/arm_grasp'

# rewards.py unchanged from v8.9 - upload same version for consistency
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


def reaching_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
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
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def reaching_bonus_open(
    env: ManagerBasedRLEnv,
    std: float,
    open_pos: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    proximity = 1 - torch.tanh(distance / std)
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_mean = gripper_pos.mean(dim=1)
    gripper_open_frac = (gripper_mean / open_pos).clamp(min=0.0, max=1.0)
    return proximity * gripper_open_frac


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
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    distance = torch.norm(obj.data.root_pos_w - ee_w, dim=1)
    near_object = (distance < threshold).float()
    gripper_pos = robot.data.joint_pos[:, -2:]
    gripper_mean = gripper_pos.mean(dim=1)
    gripper_closed = (1.0 - gripper_mean / open_pos).clamp(min=0.0, max=1.0)
    return near_object * gripper_closed


def soft_grasp_reward(
    env: ManagerBasedRLEnv,
    std: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
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
    height_gate = (obj.data.root_pos_w[:, 2] > minimal_height).float()
    return proximity * low_vel * height_gate


def cup_upright_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    dist_threshold: float,
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
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] < min_height, 1.0, 0.0)


def _compute_grasp_score(
    env: ManagerBasedRLEnv,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
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


def grasped_ee_height_reward(
    env: ManagerBasedRLEnv,
    min_height: float,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    grasp_score = _compute_grasp_score(
        env, grasp_threshold, open_pos, robot_cfg, object_cfg, ee_frame_cfg
    )
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    height_reward = ((ee_z - min_height) / 0.2).clamp(min=0.0, max=1.0)
    return grasp_score * height_reward


def grasped_ee_above_object_reward(
    env: ManagerBasedRLEnv,
    margin: float,
    scale: float,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    grasp_score = _compute_grasp_score(
        env, grasp_threshold, open_pos, robot_cfg, object_cfg, ee_frame_cfg
    )
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_z = ee_frame.data.target_pos_w[..., 0, 2]
    obj_z = obj.data.root_pos_w[:, 2]
    above = (ee_z - obj_z - margin).clamp(min=0.0)
    height_reward = (above / scale).clamp(max=1.0)
    return grasp_score * height_reward


def grasped_object_upvel_reward(
    env: ManagerBasedRLEnv,
    scale: float,
    grasp_threshold: float,
    open_pos: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Object upward velocity when grasped. Cannot be faked by static hover."""
    grasp_score = _compute_grasp_score(
        env, grasp_threshold, open_pos, robot_cfg, object_cfg, ee_frame_cfg
    )
    obj: RigidObject = env.scene[object_cfg.name]
    obj_vel_z = obj.data.root_lin_vel_w[:, 2].clamp(min=0.0)
    vel_reward = (obj_vel_z / scale).clamp(max=1.0)
    return grasp_score * vel_reward
'''

PICK_PLACE_CFG_PY = r'''"""Pick-and-Place v8.10: fix two confirmed bugs from v8.9.

Bug 1 fixed: drop_penalty threshold (0.50) was below termination (0.60).
  Reward is computed BEFORE termination check each step.
  Object drops to z=0.61 -> penalty checks 0.61 > 0.50 -> no penalty
  Then termination fires. Penalty NEVER triggered.
  Fix: drop_penalty min_height 0.50->0.72, termination 0.60->0.65

Bug 2 fixed: grasp_threshold=0.12m too large.
  At 12cm distance, 4cm-wide object cannot be physically gripped.
  All grasp-gated rewards (upvel, lifting, goal) fired with no real grasp.
  Fix: grasp_threshold 0.12->0.09m (still generous but physically meaningful)

v8.9 kept: gripper starts OPEN (0.04), velocity reward, no grasped_ee_height.
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

    robot = ARM_6DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.75)
    # v8.9: gripper starts OPEN
    robot.init_state.joint_pos = {
        "joint_1": 0.0,
        "joint_2": 0.0,
        "joint_3": 0.0,
        "joint_4": 0.0,
        "joint_5": 0.0,
        "joint_6": 0.0,
        "gripper_left_joint": 0.04,
        "gripper_right_joint": 0.04,
    }

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

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.0, 0.78), rot=(1, 0, 0, 0)),
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
                static_friction=1.5, dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.27, 0.07)),
        ),
    )

    table_high = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableHigh",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.375)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.4)),
        ),
    )

    table_low = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLow",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.275)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.55),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.50, 0.55)),
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
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",
        resampling_time_range=(5.0, 5.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.35), pos_y=(-0.15, 0.15), pos_z=(0.10, 0.30),
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
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
    """v8.10: two bug fixes on top of v8.9.

    Bug 1 fix: drop_penalty 0.50->0.72, termination 0.60->0.65
      penalty now fires BEFORE termination (table surface = 0.75)

    Bug 2 fix: grasp_threshold 0.12->0.09m
      more physically meaningful; 9cm allows correct approach geometry
    """

    reaching = RewTerm(func=mdp.reaching_penalty, weight=3.0)
    reaching_fine = RewTerm(func=mdp.reaching_bonus, params={"std": 0.3}, weight=8.0)

    grasp = RewTerm(
        func=mdp.grasp_reward,
        params={"threshold": 0.09, "open_pos": 0.04},  # 0.12 -> 0.09
        weight=3.0,
    )
    grasped_object_upvel = RewTerm(
        func=mdp.grasped_object_upvel_reward,
        params={"scale": 0.3, "grasp_threshold": 0.09, "open_pos": 0.04},  # 0.12 -> 0.09
        weight=25.0,
    )
    lifting_object = RewTerm(
        func=mdp.grasped_height_reward,
        params={
            "initial_height": 0.78, "max_bonus_height": 1.0,
            "grasp_threshold": 0.09, "open_pos": 0.04,  # 0.12 -> 0.09
        },
        weight=20.0,
    )
    object_goal_tracking = RewTerm(
        func=mdp.grasped_goal_distance,
        params={
            "std": 0.3, "command_name": "object_pose", "minimal_height": 0.80,
            "grasp_threshold": 0.09, "open_pos": 0.04,  # 0.12 -> 0.09
        },
        weight=16.0,
    )
    object_goal_tracking_fine = RewTerm(
        func=mdp.grasped_goal_distance,
        params={
            "std": 0.05, "command_name": "object_pose", "minimal_height": 0.80,
            "grasp_threshold": 0.09, "open_pos": 0.04,  # 0.12 -> 0.09
        },
        weight=5.0,
    )

    # Bug 1 fix: 0.50 -> 0.72 (fires 3cm below table surface, BEFORE termination at 0.65)
    drop_penalty = RewTerm(
        func=mdp.object_dropped_penalty,
        params={"min_height": 0.72},
        weight=-20.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.3,
        params={"asset_cfg": SceneEntityCfg(
            "robot",
            joint_names=["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
        )},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Bug 1 fix: 0.60 -> 0.65 (above drop_penalty threshold 0.72? No: 0.65 < 0.72)
    # Order: penalty fires at 0.72 (step N), termination fires at 0.65 (step N+k)
    # Both fire correctly now.
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.65, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    pass


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
        self.episode_length_s = 8.0
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

AGENTS_PPO_PY = r'''"""RSL-RL PPO configurations for arm tasks."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ArmLiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
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
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class ArmPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """v8.10: fix drop_penalty bug + grasp_threshold reduction."""
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "arm_pick_place_v8_10"
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
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.003, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-4, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )
'''


def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    print("=" * 60)
    print("  Pick-and-Place v8.10 Deployment")
    print("  Fix Bug1: drop_penalty threshold | Bug2: grasp_threshold")
    print("=" * 60)
    print("\nBug 1: drop_penalty 0.50->0.72, termination 0.60->0.65")
    print("  penalty now fires BEFORE episode ends (table surface = 0.75)")
    print("Bug 2: grasp_threshold 0.12->0.09m")
    print("  more physically meaningful for 4cm-wide object")
    print()

    import paramiko, time
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=15)
    print("Connected!")

    print("\nStopping v8.9 training...")
    client.exec_command('pkill -9 -f "train_pick_place" 2>/dev/null', timeout=5)
    time.sleep(3)

    sftp = client.open_sftp()
    files = {
        f'{BASE}/arm_grasp/mdp/rewards.py': REWARDS_PY,
        f'{BASE}/arm_grasp/envs/pick_place_cfg.py': PICK_PLACE_CFG_PY,
        f'{BASE}/arm_grasp/agents/rsl_rl_ppo_cfg.py': AGENTS_PPO_PY,
    }
    print("\nUploading files...")
    for path, content in files.items():
        with sftp.file(path, 'w') as f:
            f.write(content)
        print(f"  Uploaded: {path.split('/')[-1]}")
    sftp.close()

    client.exec_command(
        f'find {BASE}/arm_grasp -name "__pycache__" -type d -exec rm -rf {{}} + 2>/dev/null',
        timeout=5,
    )
    time.sleep(1)

    print("\nRunning validation...")
    cmd = (
        f'cd {BASE} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        '/root/autodl-tmp/conda_envs/thunder2/bin/python -u scripts/validate_pick_place.py 2>&1'
    )
    stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
    exit_code = stdout.channel.recv_exit_status()
    output = stdout.read().decode()
    idx = output.find('=== Pick-and-Place')
    print(output[idx:] if idx >= 0 else output[-2000:])
    print("Validation exit code:", exit_code)

    if exit_code != 0:
        print("\nVALIDATION FAILED!")
        client.close()
        return

    print("\nStarting v8.10 training...")
    train_cmd = (
        f'cd {BASE} && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u '
        f'scripts/train_pick_place.py --num_envs 2048 --headless > {BASE}/train_pick_place.log 2>&1 &'
    )
    client.exec_command(train_cmd, timeout=10)
    time.sleep(5)

    _, ps_out, _ = run_cmd(client, 'ps aux | grep train_pick_place | grep -v grep | head -3', timeout=5)
    if ps_out.strip():
        print("Training started!")
    else:
        _, log, _ = run_cmd(client, f'tail -20 {BASE}/train_pick_place.log', timeout=10)
        print("WARNING:", log)

    client.close()
    print("\nDone! Monitor: python check_pick_place.py")


if __name__ == "__main__":
    main()
