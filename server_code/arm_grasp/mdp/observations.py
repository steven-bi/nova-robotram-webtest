"""Custom observation functions for arm grasp tasks.

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
