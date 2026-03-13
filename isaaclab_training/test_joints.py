import argparse
import math
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# 仿真设置
sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])

# 灯光和地面
sim_utils.DomeLightCfg(intensity=500.0, color=(0.2, 0.2, 0.2)).func(
    "/World/DomeLight", sim_utils.DomeLightCfg(intensity=500.0, color=(0.2, 0.2, 0.2))
)
sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)).func(
    "/World/MainLight", sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
)
sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1)).func(
    "/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1))
)

# 加载机械臂
robot_cfg = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/root/gpufree-data/nova_training/assets/arm.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 0.0,
            "joint_4": 0.0,
            "joint_5": 0.0,
            "joint_6": 0.0,
            "gripper_left_joint": 0.0,
            "gripper_right_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit=100.0,
            velocity_limit=17.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_.*_joint"],
            effort_limit=1.0,
            velocity_limit=0.1,
            stiffness=50.0,
            damping=5.0,
        ),
    },
)

robot = Articulation(robot_cfg)
sim.reset()
robot.reset()

print("\n=== 关节信息 ===")
print(f"关节名称: {robot.joint_names}")
print(f"关节数量: {robot.num_joints}")

# 测试序列：(关节名, 目标值rad, 描述)
test_sequence = [
    ("joint_1", math.radians(30),  "J1 +30°"),
    ("joint_1", math.radians(-30), "J1 -30°"),
    ("joint_1", 0.0,               "J1 回零"),
    ("joint_2", math.radians(-20), "J2 -20°"),
    ("joint_2", 0.0,               "J2 回零"),
    ("joint_3", math.radians(20),  "J3 +20°"),
    ("joint_3", 0.0,               "J3 回零"),
    ("joint_4", math.radians(20),  "J4 +20°"),
    ("joint_4", 0.0,               "J4 回零"),
    ("joint_5", math.radians(20),  "J5 +20°"),
    ("joint_5", 0.0,               "J5 回零"),
    ("joint_6", math.radians(45),  "J6 +45°"),
    ("joint_6", 0.0,               "J6 回零"),
]

# 夹爪测试单独处理（两关节同步）
gripper_sequence = [
    (0.04, "夹爪 打开"),
    (0.0,  "夹爪 关闭"),
    (0.04, "夹爪 打开"),
    (0.0,  "夹爪 关闭"),
]

HOLD_STEPS = 150
step = 0
test_idx = 0
gripper_idx = 0
gripper_phase = False
current_targets = torch.zeros(1, robot.num_joints)

left_idx  = robot.joint_names.index("gripper_left_joint")
right_idx = robot.joint_names.index("gripper_right_joint")

print("\n=== 开始关节测试，观察每个关节方向是否与实物一致 ===\n")

while simulation_app.is_running():
    if not gripper_phase:
        if test_idx < len(test_sequence) and step % HOLD_STEPS == 0:
            joint_name, target_rad, desc = test_sequence[test_idx]
            joint_idx = robot.joint_names.index(joint_name)
            current_targets[0, joint_idx] = target_rad
            print(f"[{test_idx+1}/{len(test_sequence)}] {desc}")
            test_idx += 1
            if test_idx >= len(test_sequence):
                gripper_phase = True

    else:
        if gripper_idx < len(gripper_sequence) and step % HOLD_STEPS == 0:
            val, desc = gripper_sequence[gripper_idx]
            current_targets[0, left_idx]  = val
            current_targets[0, right_idx] = val
            print(f"[夹爪 {gripper_idx+1}/{len(gripper_sequence)}] {desc}")
            gripper_idx += 1

    robot.set_joint_position_target(current_targets)
    robot.write_data_to_sim()
    sim.step()
    robot.update(sim.cfg.dt)
    step += 1

    if gripper_phase and gripper_idx >= len(gripper_sequence) and step % HOLD_STEPS == 0:
        print("\n=== 测试完成 ===")
        break

simulation_app.close()
