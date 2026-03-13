"""Nova 机械臂 Pick-and-Place 强化学习环境 (Isaac Lab 2.x)

动作空间 (7维):
    [0:6] Δq[J1..J6]  关节增量, 每步 ±0.05 rad (≈±3°)
    [6]   gripper_cmd  夹爪指令, >0=打开, <=0=关闭

观测空间 (34维):
    [0:6]   关节角度 J1-J6, 归一化到 [-1,1]
    [6:12]  关节速度 J1-J6, 归一化
    [12:15] 末端位置 (世界坐标, m)
    [15:19] 末端四元数
    [19:22] 物体位置 (世界坐标, m)
    [22:26] 物体四元数
    [26:29] EE→物体 向量
    [29:32] 物体→目标 向量
    [32]    夹爪开合度 [0,1]
    [33]    是否抓取 {0,1}
"""
from __future__ import annotations

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# ── 路径 ────────────────────────────────────────────────────────────────────
ARM_USD_PATH = "/home/bsrl/hongsenpang/nova_training/assets/arm.usd"

# ── 场景常量 ────────────────────────────────────────────────────────────────
TABLE_HEIGHT    = 0.6          # 桌面高度 (m)
TABLE_POS_Y     = 0.20         # 桌子中心 Y 坐标 (机械臂在桌后沿 y=0，桌子向前延伸到 y=0.5)
TABLE_SIZE_X    = 0.8          # 桌面宽度 (m)
TABLE_SIZE_Y    = 0.6          # 桌面深度 (m)，Y 范围 [-0.1, 0.5]
OBJ_SPAWN_Y     = 0.32         # 物体生成 Y 坐标 (距机械臂底座 0.32m)
CYLINDER_RADIUS = 0.035        # 水杯半径 (m)
CYLINDER_HEIGHT = 0.12         # 水杯高度 (m)
OBJ_SPAWN_Z     = TABLE_HEIGHT + CYLINDER_HEIGHT / 2 + 0.002   # 物体质心高度

# 关节限位 (rad), 与实物标定值对应
JOINT_LIMITS = {
    "joint_1": (-2.618, 2.618),   # ±150°
    "joint_2": (-2.094, 0.0),     # -120°~0° (机械限位)
    "joint_3": (-2.094, 2.094),   # ±120°
    "joint_4": (-1.571, 1.571),   # ±90°
    "joint_5": (-1.571, 1.571),   # ±90°
    "joint_6": (-3.14,  3.14),    # ±180°
}

# HOME 位姿 (rad)
HOME_JOINT_POS = {
    "joint_1": 0.0017,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": -0.194,
    "joint_5": -0.024,
    "joint_6": 0.003,
    "gripper_left_joint":  0.04,  # 开启
    "gripper_right_joint": 0.04,
}


# ══════════════════════════════════════════════════════════════════════════════
# 环境配置
# ══════════════════════════════════════════════════════════════════════════════
@configclass
class NovaPickPlaceEnvCfg(DirectRLEnvCfg):

    # 仿真参数
    decimation: int = 2   # policy 25Hz = 50Hz physics / 2
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=0.02,          # 50Hz
        render_interval=2,
    )

    # 场景
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )

    # 机械臂配置
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ARM_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_HEIGHT),
            rot=(0.7071, 0.0, 0.0, 0.7071),  # 绕 Z 轴旋转 +90°，夹爪朝向 +Y（物体方向）
            joint_pos=HOME_JOINT_POS,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                effort_limit=100.0,
                velocity_limit=17.0,
                stiffness=400.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*_joint"],
                effort_limit=1.0,
                velocity_limit=0.1,
                stiffness=200.0,
                damping=20.0,
            ),
        },
    )

    # 目标物体 (圆柱体/水杯)
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            radius=CYLINDER_RADIUS,
            height=CYLINDER_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.3, 0.1),  # 橙红色
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, OBJ_SPAWN_Y, OBJ_SPAWN_Z),
        ),
    )

    # 空间维度
    action_space: int    = 7
    observation_space: int = 34
    state_space: int     = 0

    # Episode 长度 (秒)
    episode_length_s: float = 10.0

    # 奖励权重（参考 IsaacLab Franka Lift 设计）
    # ── 阶段1: 到达 ──────────────────────────────────────────────────────────
    rew_reach:          float =  2.0   # 1 - tanh(dist_ee_obj / 0.1)，引导 EE 靠近物体
    # ── 阶段2: 抬起 ──────────────────────────────────────────────────────────
    rew_lift:           float = 15.0   # 物体高度 > 桌面+4cm → binary 1.0（高权重强制学习抬起）
    # ── 阶段3: 搬运（高度门控，仅抬起后激活）────────────────────────────────
    rew_transport_c:    float = 16.0   # 1 - tanh(dist_obj_goal / 0.3)，粗粒度方向引导
    rew_transport_f:    float =  5.0   # 1 - tanh(dist_obj_goal / 0.05)，精细定位
    # ── 阶段4: 放置 ──────────────────────────────────────────────────────────
    rew_place:          float = 20.0   # dist_obj_goal < 5cm & 夹爪打开
    # ── 平滑惩罚 ─────────────────────────────────────────────────────────────
    rew_action_rate:    float = -1e-4  # 动作变化量惩罚（抑制抖动）


# ══════════════════════════════════════════════════════════════════════════════
# 环境实现
# ══════════════════════════════════════════════════════════════════════════════
class NovaPickPlaceEnv(DirectRLEnv):
    cfg: NovaPickPlaceEnvCfg

    def __init__(self, cfg: NovaPickPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 关节索引
        self._arm_ids = [
            self.robot.joint_names.index(f"joint_{i}") for i in range(1, 7)
        ]
        self._gripper_l_id = self.robot.joint_names.index("gripper_left_joint")
        self._gripper_r_id = self.robot.joint_names.index("gripper_right_joint")

        # 末端 body 索引 (link_6 为 EE 参考点)
        self._ee_id = self.robot.body_names.index("link_6")

        # 关节限位张量 (N, 6)
        lowers = [JOINT_LIMITS[f"joint_{i}"][0] for i in range(1, 7)]
        uppers = [JOINT_LIMITS[f"joint_{i}"][1] for i in range(1, 7)]
        self._q_lower = torch.tensor(lowers, device=self.device).unsqueeze(0)
        self._q_upper = torch.tensor(uppers, device=self.device).unsqueeze(0)

        # 缓冲区
        self._joint_targets = torch.zeros(
            self.num_envs, self.robot.num_joints, device=self.device
        )
        self._goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._grasp_state      = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_grasp_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._lift_rewarded    = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_actions     = torch.zeros(self.num_envs, 7, device=self.device)  # 动作平滑惩罚用

    # ── 场景构建 ──────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self.robot  = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        # 地面
        sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.15)).func(
            "/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.15))
        )

        # 桌子 (静态长方体)  机械臂底座在桌后沿 y=0，桌面向前延伸
        sim_utils.CuboidCfg(
            size=(TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.5, 0.35), metallic=0.0
            ),
        ).func(
            "/World/envs/env_.*/Table",
            sim_utils.CuboidCfg(
                size=(TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_HEIGHT),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.6, 0.5, 0.35), metallic=0.0
                ),
            ),
            translation=(0.0, TABLE_POS_Y, TABLE_HEIGHT / 2),
        )

        # 目标标记 (小绿球, 无碰撞)
        sim_utils.SphereCfg(
            radius=0.03,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.9, 0.1), metallic=0.0
            ),
        ).func(
            "/World/envs/env_.*/GoalMarker",
            sim_utils.SphereCfg(
                radius=0.03,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 0.9, 0.1), metallic=0.0
                ),
            ),
            translation=(0.0, OBJ_SPAWN_Y, OBJ_SPAWN_Z),
        )

        # 灯光
        sim_utils.DomeLightCfg(intensity=400.0, color=(0.4, 0.4, 0.4)).func(
            "/World/DomeLight",
            sim_utils.DomeLightCfg(intensity=400.0, color=(0.4, 0.4, 0.4)),
        )
        sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)).func(
            "/World/MainLight",
            sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
        )

        # 克隆并注册
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])
        self.scene.articulations["robot"]  = self.robot
        self.scene.rigid_objects["object"] = self.object

    # ── 动作处理 ──────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        # 关节增量 (每步 ±0.1 rad ≈ ±6°，加快探索)
        delta_q = self._actions[:, :6] * 0.1

        # 累积并限位
        cur_q = self._joint_targets[:, self._arm_ids[0]:self._arm_ids[-1]+1].clone()
        # 逐关节更新
        for local_i, global_i in enumerate(self._arm_ids):
            self._joint_targets[:, global_i] = torch.clamp(
                self._joint_targets[:, global_i] + delta_q[:, local_i],
                self._q_lower[:, local_i],
                self._q_upper[:, local_i],
            )

        # 夹爪指令: >0=打开(0.04m), <=0=关闭(0.0m)
        gripper_pos = torch.where(
            self._actions[:, 6] > 0,
            torch.full((self.num_envs,), 0.04, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )
        self._joint_targets[:, self._gripper_l_id] = gripper_pos
        self._joint_targets[:, self._gripper_r_id] = gripper_pos

        self.robot.set_joint_position_target(self._joint_targets)

    # ── 观测 ──────────────────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos   # (N, num_joints)
        joint_vel = self.robot.data.joint_vel

        # 归一化关节角 [-1,1]
        arm_pos = joint_pos[:, self._arm_ids[0]:self._arm_ids[-1]+1]
        arm_pos_norm = arm_pos / math.pi

        arm_vel = joint_vel[:, self._arm_ids[0]:self._arm_ids[-1]+1]
        arm_vel_norm = arm_vel / 10.0

        # 末端位置/姿态 (世界坐标)
        ee_pos  = self.robot.data.body_pos_w[:, self._ee_id, :]   # (N,3)
        ee_quat = self.robot.data.body_quat_w[:, self._ee_id, :]  # (N,4)

        # 物体位置/姿态
        obj_pos  = self.object.data.root_pos_w   # (N,3)
        obj_quat = self.object.data.root_quat_w  # (N,4)

        # 相对向量
        ee_to_obj  = obj_pos - ee_pos
        obj_to_goal = self._goal_pos - obj_pos

        # 夹爪开合度 [0,1]
        gripper_open = (
            joint_pos[:, self._gripper_l_id:self._gripper_l_id+1] / 0.04
        )

        # 是否抓取
        grasped = self._grasp_state.float().unsqueeze(-1)

        obs = torch.cat([
            arm_pos_norm,    # 6
            arm_vel_norm,    # 6
            ee_pos,          # 3
            ee_quat,         # 4
            obj_pos,         # 3
            obj_quat,        # 4
            ee_to_obj,       # 3
            obj_to_goal,     # 3
            gripper_open,    # 1
            grasped,         # 1
        ], dim=-1)           # 总计 34

        return {"policy": obs}

    # ── 奖励 ──────────────────────────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        ee_pos   = self.robot.data.body_pos_w[:, self._ee_id, :]
        obj_pos  = self.object.data.root_pos_w
        gripper_q = self.robot.data.joint_pos[:, self._gripper_l_id]

        dist_ee_obj   = torch.norm(ee_pos - obj_pos, dim=-1)
        dist_obj_goal = torch.norm(obj_pos - self._goal_pos, dim=-1)

        gripper_closed = gripper_q < 0.008          # 夹爪基本闭合
        gripper_open   = gripper_q > 0.030           # 夹爪打开（放置判定）
        obj_lifted     = obj_pos[:, 2] > (TABLE_HEIGHT + 0.04)   # 离桌 4cm

        # 更新抓取状态（放宽判定: dist < 0.08m 且夹爪部分闭合）
        self._prev_grasp_state = self._grasp_state.clone()
        newly_grasped = gripper_closed & (dist_ee_obj < 0.08)
        self._grasp_state = newly_grasped | (self._grasp_state & obj_lifted)

        # ── 阶段1: 到达 (1 - tanh, std=0.3, 全正奖励，0.5m处仍有有效梯度) ──
        r_reach = self.cfg.rew_reach * (1.0 - torch.tanh(dist_ee_obj / 0.3))

        # ── 阶段2: 抬起 (高权重二值奖励) ────────────────────────────────────
        r_lift = self.cfg.rew_lift * obj_lifted.float()

        # ── 阶段3: 搬运 (仅抬起后激活，双精度引导) ──────────────────────────
        lifted_mask = obj_lifted.float()
        r_transport_c = self.cfg.rew_transport_c * (
            1.0 - torch.tanh(dist_obj_goal / 0.3)
        ) * lifted_mask
        r_transport_f = self.cfg.rew_transport_f * (
            1.0 - torch.tanh(dist_obj_goal / 0.05)
        ) * lifted_mask

        # ── 阶段4: 放置成功 ──────────────────────────────────────────────────
        r_place = self.cfg.rew_place * (
            (dist_obj_goal < 0.05) & gripper_open
        ).float()

        # ── 平滑惩罚: 动作变化量 ─────────────────────────────────────────────
        r_action_rate = self.cfg.rew_action_rate * torch.norm(
            self._actions - self._prev_actions, dim=-1
        )
        self._prev_actions = self._actions.clone()

        return r_reach + r_lift + r_transport_c + r_transport_f + r_place + r_action_rate

    # ── 终止条件 ──────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self.object.data.root_pos_w
        dist_obj_goal = torch.norm(obj_pos - self._goal_pos, dim=-1)

        # 成功: 物体到达目标(5cm内)且夹爪打开(已放下)
        success = (dist_obj_goal < 0.05) & (
            self.robot.data.joint_pos[:, self._gripper_l_id] > 0.030
        )

        # 失败: 物体掉落桌面
        fallen = obj_pos[:, 2] < (TABLE_HEIGHT - 0.1)

        terminated = success | fallen
        truncated  = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    # ── 重置 ──────────────────────────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # 重置机械臂到 HOME
        home = torch.zeros(n, self.robot.num_joints, device=self.device)
        for jname, jval in HOME_JOINT_POS.items():
            idx = self.robot.joint_names.index(jname)
            home[:, idx] = jval
        self._joint_targets[env_ids] = home
        self.robot.set_joint_position_target(home, env_ids=env_ids)
        self.robot.write_data_to_sim()

        # 随机化物体位置 (桌面上)
        obj_xy = torch.zeros(n, 2, device=self.device).uniform_(-0.15, 0.15)
        obj_y  = obj_xy[:, 1] + OBJ_SPAWN_Y
        obj_x  = obj_xy[:, 0]
        obj_z  = torch.full((n,), OBJ_SPAWN_Z, device=self.device)
        obj_pos = torch.stack([obj_x, obj_y, obj_z], dim=-1)
        obj_quat = torch.zeros(n, 4, device=self.device)
        obj_quat[:, 0] = 1.0  # identity quaternion
        obj_pose = torch.cat([obj_pos, obj_quat], dim=-1)
        self.object.write_root_pose_to_sim(obj_pose, env_ids=env_ids)

        # 随机化目标位置 (桌面上, 与物体不重叠)
        goal_xy = torch.zeros(n, 2, device=self.device).uniform_(-0.15, 0.15)
        # 确保目标与物体距离 > 0.15m
        too_close = torch.norm(goal_xy - obj_xy, dim=-1) < 0.15
        while too_close.any():
            new_xy = torch.zeros(n, 2, device=self.device).uniform_(-0.15, 0.15)
            goal_xy[too_close] = new_xy[too_close]
            too_close = torch.norm(goal_xy - obj_xy, dim=-1) < 0.15

        self._goal_pos[env_ids, 0] = goal_xy[:, 0]
        self._goal_pos[env_ids, 1] = goal_xy[:, 1] + OBJ_SPAWN_Y
        self._goal_pos[env_ids, 2] = OBJ_SPAWN_Z

        # 重置状态标志
        self._grasp_state[env_ids]      = False
        self._prev_grasp_state[env_ids] = False
        self._lift_rewarded[env_ids]    = False
        self._prev_actions[env_ids]     = 0.0
