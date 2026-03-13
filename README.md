# NOVA Robot Arm

**inovxio** 自研 6-DOF 机械臂 —— 双臂遥操作 + Isaac Lab 强化学习训练。

## 两种控制模式

| 模式 | 路径 | 说明 | 状态 |
|------|------|------|------|
| **遥操作** | `robot_arm/grpc_stream/` | 主从双臂跟随，100Hz，MIT 力矩控制 | ✅ 真机已接通 |
| **RL 自主** | `server_code/arm_grasp/` | Isaac Lab 训练 → ONNX → 自主抓取 | 🔵 训练中 |

### 遥操作架构

```
主臂（client.py）                    从臂（server.py）
  │ 读取 6 关节 pos/vel/torque           │ 接收主臂关节状态
  │ 逆动力学计算力矩补偿                  │ 驱动从臂跟随运动
  └──── gRPC 流式推送 100Hz ────────────►│ Robstride CAN 电机
        10.16.77.87:50051                │ MIT 模式（Kp/Kd 可调）
```

---

## 项目结构

```
robotarm_ws/
├── robot_arm/
│   ├── grpc_stream/            ← 真机遥操作控制
│   │   ├── server.py           ← 从臂：接收指令，驱动电机（跑在从臂主控）
│   │   ├── client.py           ← 主臂：读取状态，发送指令（跑在主臂主控）
│   │   ├── robstride.py        ← Robstride CAN 总线电机驱动
│   │   ├── Inv_Dyn.py          ← 逆动力学（关节力矩补偿）
│   │   └── Interpolation.py    ← 关节插值
│   ├── kinematics/             ← 逆运动学求解
│   └── verification/           ← 运动学验证 + MuJoCo Demo 视频
├── server_code/arm_grasp/      ← Isaac Lab RL 自主训练包
│   ├── envs/
│   │   ├── lift_cube_cfg.py    ← Task 1: 抬起方块
│   │   └── pick_place_cfg.py   ← Task 2: Pick & Place（当前 v8.10）
│   ├── mdp/rewards.py          ← 奖励函数
│   └── agents/rsl_rl_ppo_cfg.py← PPO 训练配置
├── arm/                        ← ROS2 包（URDF/launch/meshes）
├── src/pkg_robotarm_py/        ← Python ROS2 控制包
├── arm_updated.urdf            ← 机械臂 URDF
└── scripts/                    ← 工具脚本（训练部署/回放/录制）
    ├── start_training.py       ← 启动 PPO 训练到 AutoDL 服务器
    ├── deploy.py               ← 部署最新训练代码（v8.10）
    ├── play_script.py          ← 仿真回放
    └── archive/                ← 历史迭代版本（v8.5~v8.9）
```

---

## 快速开始

### 遥操作（真机）

```bash
# 1. 从臂主控上启动 server（等待主臂连接）
python robot_arm/grpc_stream/server.py

# 2. 主臂主控上启动 client（开始遥操作）
python robot_arm/grpc_stream/client.py
```

### RL 训练（Isaac Lab）

```bash
# 部署并启动训练到 AutoDL 服务器
python scripts/start_training.py

# 仿真回放已训练策略
python scripts/play_script.py
```

---

## 训练任务进展

| 任务 | 环境 | 奖励版本 | 状态 |
|------|------|---------|------|
| Lift Cube | `ArmLiftCubeEnvCfg` | — | ✅ 完成 |
| Pick & Place | `ArmPickPlaceEnvCfg` | v8.10 | 🔵 训练中 |

**v8.10 修复记录**：
- `drop_penalty` 阈值低于终止阈值 → 惩罚从不触发，已修复
- `grasp_threshold=0.12m` 过大（4cm 夹爪无法在 12cm 距离抓住物体）→ 改为 0.09m

---

## 硬件规格

- **电机**：Robstride 关节电机（CAN 总线，`ControlCAN.dll`）
- **控制模式**：MIT（位置 + 速度 + 力矩混合）
- **自由度**：6-DOF + 1-DOF 夹爪
- **控制频率**：100Hz
- **通信**：gRPC + Protobuf（`robot_control.proto`）

## 训练服务器

AutoDL RTX 5090：`connect.westd.seetacloud.com:14918`
路径：`/root/autodl-tmp/arm_grasp/`

## 相关项目

- [NOVA Dog](../nova-dog) — 机器狗整机（Fire Demo 中与机械臂协作）
- [AME-2](../../ame2_standalone) — 四足运动控制 RL
- [OTA 系统](../../infra/ota) — 固件远程升级
