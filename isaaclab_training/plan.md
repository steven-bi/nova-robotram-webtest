# Nova Robot Arm — Isaac Lab 抓取训练计划

> 目标：将 Nova 6-DOF+夹爪机械臂导入 Isaac Lab，训练 pick-and-place 强化学习策略，最终部署到实物。

---

## 服务器信息

| 项目 | 值 |
|------|----|
| 地址 | `fe91fae6a6756695.natapp.cc:12346` |
| 用户 | `bsrl` / `<SSH_PASS env var>` |
| GPU | 8× RTX 3090 (24GB each)，可用：GPU 4/5/6 |
| Python | `thunder2` conda env (Python 3.11) |
| Isaac Sim | `isaacsim 5.0.0` (pip package) |
| Isaac Lab | `0.46.2`，源码于 `/home/bsrl/hongsenpang/RLbased/IsaacLab/` |
| 训练目录 | `/home/bsrl/hongsenpang/nova_training/` |

### 启动命令

```bash
export CUDA_VISIBLE_DEVICES=4,5,6
export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2
export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab
bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/train.py --headless
```

---

## 阶段总览

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 资产准备（URDF → USD） | ✅ 完成 |
| Phase 2 | RL 环境基础验证（PPO） | ✅ 完成（PPO 无法收敛，已确认根因） |
| Phase 3 | **PPO-HER 方案重构** | 🔄 进行中 |
| Phase 4 | Sim2Real 部署 | ⬜ 待定 |

---

## Phase 1：资产准备 ✅

- USD：`/home/bsrl/hongsenpang/nova_training/assets/arm.usd`（已转换，含网格）
- 场景：桌面 0.6m，机械臂在桌后沿 y=0，物体在 y=0.32m 前方
- 机械臂旋转：绕 Z 轴 +90°（夹爪朝向 +Y，正对物体）

---

## Phase 2：PPO 失败根因总结 ✅

经过多轮调参（3000 轮 × 4次），PPO 始终无法收敛，根因确认如下：

| 根因 | 说明 |
|------|------|
| **稀疏奖励 + PPO 探索不足** | HOME 位置离物体 0.5~0.8m，随机动作极少碰到物体 |
| **奖励梯度消失** | `1 - tanh(d/0.1)` 在 0.5m 处梯度 ≈ 0，策略无方向感 |
| **熵坍塌** | `entropy_coef=0.005` 太低，noise_std 从 1.0 → 0.07，探索停止 |
| **坐标系问题** | 观测使用世界坐标绝对值，策略需同时学习自身位置和物体位置 |
| **无记忆机制** | PPO 是 on-policy，失败轨迹直接丢弃，无法从"差一点成功"中学习 |

**结论：单纯 PPO 不适合稀疏奖励的操作任务。**

---

## Phase 3：PPO-HER 方案重构 🔄

### 核心思路

**Hindsight Experience Replay（HER）**：
将失败轨迹重新打标签——把"手臂最终停在哪里"标记为"目标"，重新计算奖励，失败轨迹变成成功轨迹。
论文（arXiv:2410.22524）证明：PPO-HER 在操作任务中收敛速度优于 SAC-HER。

```
普通 PPO：
  失败轨迹 → 直接丢弃 → 没有学到任何东西

PPO-HER：
  失败轨迹 → 重标签（把末端位置设为目标）→ 变成"成功"轨迹
           → PPO 从中学习"如何到达任意位置"
           → 再逐步学习"如何到达真正的目标"
```

### 同步改动（3项，与 HER 配合）

#### 改动1：观测空间改为机器人相对坐标系

```python
# 当前（世界坐标）：策略需要学"我在哪 + 物体在哪"
ee_pos  = (0.12, 0.85, 1.1)   # 每次不同，难学
obj_pos = (0.08, 0.32, 0.66)

# 改后（相对坐标）：策略只需学"物体相对我在哪"
ee_to_obj_vec   = obj_pos - ee_pos    # (3,) 相对向量
obj_to_goal_vec = goal - obj_pos      # (3,) 相对向量
```

观测维度从 34 维重新设计为 **goal-conditioned** 格式：

```
observation (25维):
  [0:6]   关节角 J1-J6（归一化）
  [6:12]  关节速度（归一化）
  [12:15] EE → 物体 相对向量（机器人局部坐标系）
  [15:18] 物体 → 目标 相对向量
  [18:21] EE 在机器人基座坐标系中的位置
  [21]    夹爪开合度 [0,1]
  [22]    是否抓取 {0,1}

achieved_goal (3维):
  物体当前位置（世界坐标，用于 HER 重标签）

desired_goal (3维):
  目标放置位置（世界坐标，用于 HER 重标签）
```

#### 改动2：动作空间改为关节速度

```python
# 当前：关节位置增量（累积）
delta_q = action * 0.1  # rad/step

# 改后：关节速度（直接映射）
joint_vel_target = action * max_vel  # rad/s
```

研究表明关节速度比位置增量有更好的 sim2real 迁移性。

#### 改动3：渐进课程（Curriculum Reset）

```python
# 前 500 轮：50% 概率初始化手臂在物体旁边（5~15cm）
# 500~1500 轮：20% 概率近距离初始化
# 1500 轮后：全部正常 HOME 初始化
```

让策略先学会"抓到之后怎么做"，再学"如何靠近"。

---

### 实施步骤

#### Step 1：实现 HER replay buffer
- 文件：`envs/her_buffer.py`
- 实现标准 HER（Future 策略：每条轨迹取后续 k=4 个状态作为重标签目标）
- 与 RSL-RL PPO runner 集成（覆盖 `collect_rollouts`）

#### Step 2：改写环境为 goal-conditioned 格式
- 文件：`envs/nova_pick_place_env.py` 重构
- 观测改为相对坐标系
- 增加 `achieved_goal` / `desired_goal` 字段
- 加入 Curriculum reset 逻辑

#### Step 3：修改训练脚本
- 文件：`scripts/train_her.py`（新建，保留原 `train.py` 不动）
- 集成 HER buffer
- 超参数：`entropy_coef=0.01`，`lr=3e-4`，`num_steps_per_env=64`

#### Step 4：冒烟测试
- 64 envs × 100 轮，验证 HER 重标签逻辑正确（reward 应该迅速出现正值）

#### Step 5：正式训练
- 1024 envs × 5000 轮（HER 样本效率高，不需要很多轮次）
- 验收：1000 轮内 mean reward > 5，success rate > 30%

#### Step 6：录制评估视频
- 加载最优 checkpoint，`eval_video.py` 录 30 秒

---

### 验收标准（Phase 3）

| 指标 | 目标值 |
|------|--------|
| 500 轮内 mean reward | > 2.0（明显上升趋势） |
| 1000 轮 success rate | > 20% |
| 3000 轮 success rate | > 60% |
| noise_std 趋势 | 从 1.0 逐渐下降（不崩溃） |

---

### 技术参考

| 来源 | 关键发现 |
|------|---------|
| arXiv:2410.22524 | PPO-HER 比 SAC-HER 在操作任务中收敛更快 |
| IsaacGymEnvs FrankaCubeStack | 阶段门控奖励 + tanh shaping |
| Gymnasium Robotics Fetch | goal-conditioned 观测标准格式 |
| arXiv:2312.03673 | 关节速度 > 位置增量的 sim2real 迁移 |
| arXiv:2405.00662 | PPO 失败是表示坍塌，不是熵坍塌 |

---

## Phase 4：Sim2Real 部署 ⬜

待 Phase 3 完成后规划。

```
RL Policy 输出 (joint_vel_cmd, gripper_cmd)
    ↓
关节速度 → 位置增量转换（×dt）
    ↓
joint_gui_web.py: execute_move(targets_deg)
    ↓
MIT 模式命令 → Robstride 电机
```

---

## 服务器文件结构

```
/home/bsrl/hongsenpang/nova_training/
├── assets/
│   ├── arm.usd              ✅ 已生成
│   └── meshes/              ✅ 9个 STL
├── envs/
│   ├── __init__.py
│   ├── nova_pick_place_env.py   （当前版本，PPO用，待重构）
│   └── her_buffer.py            ⬜ 待实现
├── scripts/
│   ├── train.py                 ✅（PPO版，保留参考）
│   ├── train_her.py             ⬜ 待实现
│   ├── record_video.py          ✅
│   └── eval_video.py            ✅
├── logs/
│   └── nova_pick_place/         ✅ 历史 checkpoint
└── video/                       ✅ 历史录像
```

---

*最后更新：2026-03-13*
