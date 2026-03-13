"""Nova 机械臂 Goal-Conditioned Pick-and-Place 环境 (PPO-HER 版本)

核心改进（相比原 nova_pick_place_env.py）：

1. 观测空间：改为机器人相对坐标系（去掉世界绝对坐标）
2. HER Reset：episode 结束时 50% 概率把上次物体终止位置设为新目标
3. 课程学习：前期物体在更近距离生成，逐步扩展到正常范围
4. 奖励：goal-conditioned，适配 HER 重标签逻辑

动作空间 (7维):
    [0:6] Δq[J1..J6]  关节增量, 每步 ±0.1 rad (≈±6°)
    [6]   gripper_cmd  夹爪指令, >0=打开, <=0=关闭

观测空间 (23维):
    [0:6]   关节角度 J1-J6, 归一化到 [-1,1]
    [6:12]  关节速度 J1-J6, 归一化
    [12:15] EE → 物体 相对向量 (m)        ← 替代绝对坐标
    [15:18] 物体 → 目标 相对向量 (m)      ← 替代绝对坐标
    [18:21] EE 相对机械臂底座的位置 (m)
    [21]    夹爪开合度 [0,1]
    [22]    是否抓取 {0,1}

achieved_goal (3维):  物体当前位置（用于 HER 重标签）
desired_goal  (3维):  目标放置位置（用于 HER 重标签）
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

from .her_buffer import her_reward_fn

# ── 路径 ────────────────────────────────────────────────────────────────────
ARM_USD_PATH = "/home/bsrl/hongsenpang/nova_training/assets/arm.usd"

# ── 场景常量 ────────────────────────────────────────────────────────────────
TABLE_HEIGHT    = 0.6
TABLE_POS_Y     = 0.20
TABLE_SIZE_X    = 0.8
TABLE_SIZE_Y    = 0.6
OBJ_SPAWN_Y     = 0.32          # 物体生成中心 Y 坐标
CYLINDER_RADIUS = 0.035
CYLINDER_HEIGHT = 0.12
OBJ_SPAWN_Z     = TABLE_HEIGHT + CYLINDER_HEIGHT / 2 + 0.002

# 机械臂底座在世界坐标系中的位置（fix_root_link=True，固定不动）
BASE_POS = torch.tensor([0.0, 0.0, TABLE_HEIGHT])

# 关节限位
JOINT_LIMITS = {
    "joint_1": (-2.618, 2.618),
    "joint_2": (-2.094, 0.0),
    "joint_3": (-2.094, 2.094),
    "joint_4": (-1.571, 1.571),
    "joint_5": (-1.571, 1.571),
    "joint_6": (-3.14,  3.14),
}

# HOME 位姿
HOME_JOINT_POS = {
    "joint_1": 0.0017,
    "joint_2": 0.0,
    "joint_3": 0.0,
    "joint_4": -0.194,
    "joint_5": -0.024,
    "joint_6": 0.003,
    "gripper_left_joint":  0.04,
    "gripper_right_joint": 0.04,
}

# 课程学习：物体生成范围（前期更近，后期扩展到正常值）
# 通过 env.curriculum_step 外部控制，每 500 轮更新一次
CURRICULUM_STAGES = [
    # (obj_spawn_y_center, xy_noise_range)
    (0.20, 0.05),   # Stage 0: 近距离，±5cm 噪声
    (0.25, 0.08),   # Stage 1
    (0.28, 0.10),   # Stage 2
    (0.32, 0.15),   # Stage 3: 正常（原始设置）
]


# ══════════════════════════════════════════════════════════════════════════════
# 环境配置
# ══════════════════════════════════════════════════════════════════════════════
@configclass
class NovaHEREnvCfg(DirectRLEnvCfg):

    decimation: int = 2
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(dt=0.02, render_interval=2)

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5)

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
            rot=(0.7071, 0.0, 0.0, 0.7071),
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

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            radius=CYLINDER_RADIUS,
            height=CYLINDER_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.3, 0.1), metallic=0.0
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, OBJ_SPAWN_Y, OBJ_SPAWN_Z),
        ),
    )

    # 观测 / 动作维度
    action_space: int      = 7
    observation_space: int = 23
    state_space: int       = 0

    episode_length_s: float = 12.0   # 稍长，给策略更多时间

    # ── 奖励权重 ────────────────────────────────────────────────────────────
    rew_reach:       float = 3.0    # 1 - tanh(dist_ee_obj / 0.3)，全程正奖励
    rew_lift:        float = 15.0   # 物体抬起 4cm 以上（二值）
    rew_transport_c: float = 16.0   # 粗粒度搬运（抬起后）
    rew_transport_f: float =  5.0   # 精细搬运
    rew_place:       float = 20.0   # 放置成功
    rew_action_rate: float = -1e-4  # 动作变化量惩罚

    # ── HER 参数 ─────────────────────────────────────────────────────────────
    her_prob:        float = 0.5    # episode 结束后使用 HER 目标的概率
    her_success_thresh: float = 0.05  # 判定成功的距离阈值 (m)

    # ── 课程阶段（由训练脚本外部更新） ──────────────────────────────────────
    curriculum_stage: int = 0       # 0~3，对应 CURRICULUM_STAGES


# ══════════════════════════════════════════════════════════════════════════════
# 环境实现
# ══════════════════════════════════════════════════════════════════════════════
class NovaHEREnv(DirectRLEnv):
    cfg: NovaHEREnvCfg

    def __init__(self, cfg: NovaHEREnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 关节索引
        self._arm_ids = [self.robot.joint_names.index(f"joint_{i}") for i in range(1, 7)]
        self._gripper_l_id = self.robot.joint_names.index("gripper_left_joint")
        self._gripper_r_id = self.robot.joint_names.index("gripper_right_joint")
        self._ee_id = self.robot.body_names.index("link_6")

        # 关节限位张量
        lowers = [JOINT_LIMITS[f"joint_{i}"][0] for i in range(1, 7)]
        uppers = [JOINT_LIMITS[f"joint_{i}"][1] for i in range(1, 7)]
        self._q_lower = torch.tensor(lowers, device=self.device).unsqueeze(0)
        self._q_upper = torch.tensor(uppers, device=self.device).unsqueeze(0)

        # 底座位置（固定，广播用）
        self._base_pos = BASE_POS.to(self.device).unsqueeze(0).expand(self.num_envs, -1)

        # 缓冲区
        self._joint_targets  = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self._goal_pos        = torch.zeros(self.num_envs, 3, device=self.device)
        self._grasp_state     = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_grasp_state = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._lift_rewarded   = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._prev_actions    = torch.zeros(self.num_envs, 7, device=self.device)

        # HER：记录每个 env 上一 episode 的物体终止位置
        self._last_obj_pos    = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_obj_pos[:, 1] = OBJ_SPAWN_Y   # 初始填默认值，避免第一次 HER 目标在原点
        self._last_obj_pos[:, 2] = OBJ_SPAWN_Z

    # ── 场景构建 ──────────────────────────────────────────────────────────────
    def _setup_scene(self):
        self.robot  = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.15)).func(
            "/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.15))
        )

        _table_cfg = sim_utils.CuboidCfg(
            size=(TABLE_SIZE_X, TABLE_SIZE_Y, TABLE_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.5, 0.35)),
        )
        _table_cfg.func(
            "/World/envs/env_.*/Table", _table_cfg,
            translation=(0.0, TABLE_POS_Y, TABLE_HEIGHT / 2),
        )

        _goal_cfg = sim_utils.SphereCfg(
            radius=0.03,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.1)),
        )
        _goal_cfg.func(
            "/World/envs/env_.*/GoalMarker", _goal_cfg,
            translation=(0.0, OBJ_SPAWN_Y, OBJ_SPAWN_Z),
        )

        sim_utils.DomeLightCfg(intensity=400.0).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=400.0))
        sim_utils.DistantLightCfg(intensity=2500.0).func("/World/MainLight", sim_utils.DistantLightCfg(intensity=2500.0))

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])
        self.scene.articulations["robot"]  = self.robot
        self.scene.rigid_objects["object"] = self.object

    # ── 动作 ──────────────────────────────────────────────────────────────────
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        delta_q = self._actions[:, :6] * 0.1   # ±0.1 rad/step

        for local_i, global_i in enumerate(self._arm_ids):
            self._joint_targets[:, global_i] = torch.clamp(
                self._joint_targets[:, global_i] + delta_q[:, local_i],
                self._q_lower[:, local_i],
                self._q_upper[:, local_i],
            )

        gripper_pos = torch.where(
            self._actions[:, 6] > 0,
            torch.full((self.num_envs,), 0.04, device=self.device),
            torch.zeros(self.num_envs, device=self.device),
        )
        self._joint_targets[:, self._gripper_l_id] = gripper_pos
        self._joint_targets[:, self._gripper_r_id] = gripper_pos

        self.robot.set_joint_position_target(self._joint_targets)

    # ── 观测（相对坐标系）────────────────────────────────────────────────────
    def _get_observations(self) -> dict:
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        arm_pos_norm = joint_pos[:, self._arm_ids[0]:self._arm_ids[-1]+1] / math.pi
        arm_vel_norm = joint_vel[:, self._arm_ids[0]:self._arm_ids[-1]+1] / 10.0

        ee_pos  = self.robot.data.body_pos_w[:, self._ee_id, :]
        obj_pos = self.object.data.root_pos_w

        # ── 相对向量（替代世界绝对坐标）──────────────────────────────────────
        ee_to_obj   = obj_pos  - ee_pos             # (N, 3)
        obj_to_goal = self._goal_pos - obj_pos      # (N, 3)

        # EE 相对底座的局部位置（底座固定，相当于机器人坐标系）
        ee_local = ee_pos - self._base_pos          # (N, 3)

        gripper_open = joint_pos[:, self._gripper_l_id:self._gripper_l_id+1] / 0.04
        grasped      = self._grasp_state.float().unsqueeze(-1)

        obs = torch.cat([
            arm_pos_norm,   # 6
            arm_vel_norm,   # 6
            ee_to_obj,      # 3
            obj_to_goal,    # 3
            ee_local,       # 3
            gripper_open,   # 1
            grasped,        # 1
        ], dim=-1)          # 共 23 维

        return {
            "policy":         obs,
            "achieved_goal":  obj_pos.clone(),           # HER 重标签用
            "desired_goal":   self._goal_pos.clone(),    # HER 重标签用
        }

    # ── 奖励（goal-conditioned）──────────────────────────────────────────────
    def _get_rewards(self) -> torch.Tensor:
        ee_pos    = self.robot.data.body_pos_w[:, self._ee_id, :]
        obj_pos   = self.object.data.root_pos_w
        gripper_q = self.robot.data.joint_pos[:, self._gripper_l_id]

        dist_ee_obj   = torch.norm(ee_pos - obj_pos, dim=-1)
        dist_obj_goal = torch.norm(obj_pos - self._goal_pos, dim=-1)

        gripper_closed = gripper_q < 0.008
        gripper_open   = gripper_q > 0.030
        obj_lifted     = obj_pos[:, 2] > (TABLE_HEIGHT + 0.04)

        self._prev_grasp_state = self._grasp_state.clone()
        newly_grasped = gripper_closed & (dist_ee_obj < 0.08)
        self._grasp_state = newly_grasped | (self._grasp_state & obj_lifted)

        # 阶段1: 到达（std=0.3，0.5m 处仍有有效梯度，全正奖励）
        r_reach = self.cfg.rew_reach * (1.0 - torch.tanh(dist_ee_obj / 0.3))

        # 阶段2: 抬起
        r_lift = self.cfg.rew_lift * obj_lifted.float()

        # 阶段3: 搬运（高度门控）
        lifted = obj_lifted.float()
        r_transport_c = self.cfg.rew_transport_c * (1.0 - torch.tanh(dist_obj_goal / 0.3)) * lifted
        r_transport_f = self.cfg.rew_transport_f * (1.0 - torch.tanh(dist_obj_goal / 0.05)) * lifted

        # 阶段4: 放置
        r_place = self.cfg.rew_place * ((dist_obj_goal < 0.05) & gripper_open).float()

        # 平滑惩罚
        r_action_rate = self.cfg.rew_action_rate * torch.norm(self._actions - self._prev_actions, dim=-1)
        self._prev_actions = self._actions.clone()

        return r_reach + r_lift + r_transport_c + r_transport_f + r_place + r_action_rate

    # ── 终止条件 ──────────────────────────────────────────────────────────────
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self.object.data.root_pos_w
        dist_obj_goal = torch.norm(obj_pos - self._goal_pos, dim=-1)

        success = (dist_obj_goal < self.cfg.her_success_thresh) & (
            self.robot.data.joint_pos[:, self._gripper_l_id] > 0.030
        )
        fallen = obj_pos[:, 2] < (TABLE_HEIGHT - 0.1)

        terminated = success | fallen
        truncated  = self.episode_length_buf >= self.max_episode_length
        return terminated, truncated

    # ── 重置（含 HER 目标 + 课程）────────────────────────────────────────────
    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        n = len(env_ids)

        # ── 1. 保存上次物体终止位置（供 HER 使用）────────────────────────
        self._last_obj_pos[env_ids] = self.object.data.root_pos_w[env_ids].clone()

        # ── 2. 重置机械臂到 HOME ──────────────────────────────────────────
        home = torch.zeros(n, self.robot.num_joints, device=self.device)
        for jname, jval in HOME_JOINT_POS.items():
            idx = self.robot.joint_names.index(jname)
            home[:, idx] = jval
        self._joint_targets[env_ids] = home
        self.robot.set_joint_position_target(home, env_ids=env_ids)
        self.robot.write_data_to_sim()

        # ── 3. 课程：根据当前阶段确定物体生成范围 ─────────────────────────
        stage = min(self.cfg.curriculum_stage, len(CURRICULUM_STAGES) - 1)
        spawn_y_center, noise_range = CURRICULUM_STAGES[stage]

        obj_xy  = torch.zeros(n, 2, device=self.device).uniform_(-noise_range, noise_range)
        obj_y   = obj_xy[:, 1] + spawn_y_center
        obj_x   = obj_xy[:, 0]
        obj_z   = torch.full((n,), OBJ_SPAWN_Z, device=self.device)
        obj_pos = torch.stack([obj_x, obj_y, obj_z], dim=-1)
        obj_quat = torch.zeros(n, 4, device=self.device)
        obj_quat[:, 0] = 1.0
        self.object.write_root_pose_to_sim(
            torch.cat([obj_pos, obj_quat], dim=-1), env_ids=env_ids
        )

        # ── 4. 目标位置：HER（50%）或正常随机（50%）────────────────────────
        her_mask   = torch.rand(n, device=self.device) < self.cfg.her_prob
        normal_mask = ~her_mask

        # HER 目标：上次物体终止位置（确保在桌面高度）
        her_goal = self._last_obj_pos[env_ids].clone()
        her_goal[:, 2] = OBJ_SPAWN_Z  # 保持高度合理

        # 正常目标：随机，但保证与物体位置距离 > 0.15m
        goal_xy   = torch.zeros(n, 2, device=self.device).uniform_(-noise_range, noise_range)
        too_close = torch.norm(goal_xy - obj_xy, dim=-1) < 0.15
        for _ in range(10):  # 最多重采 10 次
            if not too_close.any():
                break
            new_xy = torch.zeros(n, 2, device=self.device).uniform_(-noise_range, noise_range)
            goal_xy[too_close] = new_xy[too_close]
            too_close = torch.norm(goal_xy - obj_xy, dim=-1) < 0.15

        normal_goal = torch.stack([
            goal_xy[:, 0],
            goal_xy[:, 1] + spawn_y_center,
            torch.full((n,), OBJ_SPAWN_Z, device=self.device),
        ], dim=-1)

        # 合并
        self._goal_pos[env_ids] = torch.where(
            her_mask.unsqueeze(-1).expand(-1, 3),
            her_goal,
            normal_goal,
        )

        # ── 5. 状态标志清零 ───────────────────────────────────────────────
        self._grasp_state[env_ids]      = False
        self._prev_grasp_state[env_ids] = False
        self._lift_rewarded[env_ids]    = False
        self._prev_actions[env_ids]     = 0.0
