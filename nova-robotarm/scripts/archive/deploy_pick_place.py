"""Deploy pick-and-place environment to server and start training."""
import paramiko
import time

SSH_HOST = 'connect.westd.seetacloud.com'
SSH_PORT = 14918
SSH_USER = 'root'
SSH_PASS = 'UvUnT2x1jsaa'
BASE = '/root/autodl-tmp/arm_grasp'

# ==============================================================================
# FILE 1: arm_grasp/mdp/rewards.py (REPLACE - keep old + add new functions)
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


# ---- Existing reward functions (lift-cube) ----

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
    """Reward for closing the gripper when near the object."""
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


# ---- New reward functions (pick-and-place) ----

def place_reward(
    env: ManagerBasedRLEnv,
    std: float,
    vel_threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for placing object near target with low velocity (gentle placement)."""
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    proximity = 1 - torch.tanh(distance / std)
    # Low velocity bonus
    obj_vel = torch.norm(obj.data.root_lin_vel_w, dim=1)
    low_vel = (obj_vel < vel_threshold).float()
    return proximity * low_vel


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
    # Cup orientation: z-axis should be up
    # quat format: (w, x, y, z) -> upright means w^2+z^2-x^2-y^2 ~ 1
    quat = obj.data.root_quat_w  # (N, 4)
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
'''

# ==============================================================================
# FILE 2: arm_grasp/mdp/observations.py (REPLACE - add cup orientation)
# ==============================================================================
OBSERVATIONS_PY = r'''"""Custom observation functions for arm grasp tasks."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
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
'''

# ==============================================================================
# FILE 3: arm_grasp/mdp/__init__.py (REPLACE)
# ==============================================================================
MDP_INIT_PY = r'''"""Custom MDP functions for arm grasp tasks."""
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
'''

# ==============================================================================
# FILE 4: arm_grasp/envs/pick_place_cfg.py (NEW - core environment config)
# ==============================================================================
PICK_PLACE_CFG_PY = r'''"""Pick-and-Place environment: move coffee cup from high table to low table."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

    # Robot arm (same as lift-cube)
    robot = ARM_6DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.75)

    # End-effector frame tracker
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

    # Coffee cup (cylinder)
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.82),
            rot=(1, 0, 0, 0),
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.10,
            axis="Z",
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.27, 0.07),  # Brown coffee cup
            ),
        ),
    )

    # High table (source - same as lift-cube)
    table_high = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableHigh",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.75),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.5, 0.4),  # Wood color
            ),
        ),
    )

    # Low table (target)
    table_low = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLow",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.6, 0.55),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.50, 0.55),  # Blue-gray
            ),
        ),
    )

    # Ground plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=GroundPlaneCfg(),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Target placement position on the low table."""
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_6",
        resampling_time_range=(12.0, 12.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.40, 0.65),     # Low table center in robot body frame
            pos_y=(-0.12, 0.12),
            pos_z=(-0.20, -0.10),   # Below robot base (low table surface)
            roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Same as lift-cube: 6-DOF arm + binary gripper."""
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
        object_orientation = ObsTerm(func=mdp.object_orientation_in_robot_root_frame)
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
    # Phase 1: Reach the cup
    reaching = RewTerm(func=mdp.reaching_penalty, weight=2.0)
    reaching_fine = RewTerm(
        func=mdp.reaching_bonus, params={"std": 0.05}, weight=5.0
    )
    # Phase 2: Grasp the cup
    grasp = RewTerm(
        func=mdp.grasp_reward,
        params={"threshold": 0.08, "open_pos": 0.04},
        weight=5.0,
    )
    # Phase 3: Lift from high table
    object_height = RewTerm(
        func=mdp.object_height_reward,
        params={"initial_height": 0.80, "max_bonus_height": 1.05},
        weight=8.0,
    )
    lifting_object = RewTerm(
        func=mdp.object_is_lifted, params={"minimal_height": 0.88}, weight=5.0
    )
    # Phase 4: Transport to low table
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.5, "minimal_height": 0.40, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.1, "minimal_height": 0.40, "command_name": "object_pose"},
        weight=5.0,
    )
    # Phase 5: Gentle placement
    place_gentle = RewTerm(
        func=mdp.place_reward,
        params={"std": 0.1, "vel_threshold": 0.1, "command_name": "object_pose"},
        weight=10.0,
    )
    cup_upright = RewTerm(
        func=mdp.cup_upright_reward,
        params={"command_name": "object_pose", "dist_threshold": 0.15},
        weight=3.0,
    )
    # Penalties
    drop_penalty = RewTerm(
        func=mdp.object_dropped_penalty,
        params={"min_height": 0.20},
        weight=-10.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.10, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000},
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000},
    )


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
        self.episode_length_s = 12.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
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
# FILE 5: arm_grasp/envs/__init__.py (REPLACE)
# ==============================================================================
ENVS_INIT_PY = r'''"""Environment configurations."""
from .lift_cube_cfg import ArmLiftCubeEnvCfg, ArmLiftCubeEnvCfg_PLAY, ArmLiftSceneCfg
from .pick_place_cfg import ArmPickPlaceEnvCfg, ArmPickPlaceEnvCfg_PLAY
'''

# ==============================================================================
# FILE 6: arm_grasp/agents/rsl_rl_ppo_cfg.py (REPLACE - add new config)
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
    """PPO config for pick-and-place task (harder, longer)."""
    num_steps_per_env = 24
    max_iterations = 8000
    save_interval = 500
    experiment_name = "arm_pick_place"
    run_name = ""
    resume = False
    logger = "tensorboard"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
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
'''

# ==============================================================================
# FILE 7: arm_grasp/agents/__init__.py (REPLACE)
# ==============================================================================
AGENTS_INIT_PY = r'''"""Agent configurations."""
from .rsl_rl_ppo_cfg import ArmLiftCubePPORunnerCfg, ArmPickPlacePPORunnerCfg
'''

# ==============================================================================
# FILE 8: scripts/train_pick_place.py (NEW)
# ==============================================================================
TRAIN_SCRIPT = r'''"""Train pick-and-place policy with RSL-RL PPO."""
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
# FILE 9: scripts/validate_pick_place.py (NEW)
# ==============================================================================
VALIDATE_SCRIPT = r'''"""Validate pick-and-place environment loads and runs."""
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

print("=== Pick-and-Place Environment Validation ===")
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
    print("Connecting to SeetaCloud server...")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=15)
    print("Connected!")

    # Kill any running training
    print("Stopping any running training...")
    client.exec_command('pkill -9 -f "train"', timeout=5)
    time.sleep(2)

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

    for path, content in files.items():
        with sftp.file(path, 'w') as f:
            f.write(content)
        fname = path.split('/')[-1]
        print(f"  Uploaded: {fname}")

    sftp.close()
    print("All files uploaded!")

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

    # Show relevant output
    idx = output.find('=== Pick-and-Place')
    if idx >= 0:
        print(output[idx:])
    else:
        # Show last 3000 chars for debugging
        print(output[-3000:])
    print("Validation exit code:", exit_code)

    if exit_code != 0:
        print("\nVALIDATION FAILED! Fix errors before training.")
        client.close()
        return

    # Start training
    print("\nStarting pick-and-place training (8000 iterations)...")
    train_cmd = (
        f'cd {BASE} && '
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u '
        'scripts/train_pick_place.py --num_envs 2048 --headless '
        f'> {BASE}/train_pick_place.log 2>&1 &'
    )
    client.exec_command(train_cmd, timeout=10)
    time.sleep(3)

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
    print("\nDone! Monitor with: check_pick_place_training.py")


if __name__ == "__main__":
    main()
