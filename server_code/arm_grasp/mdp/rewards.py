"""Custom reward functions for arm grasp tasks."""
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
