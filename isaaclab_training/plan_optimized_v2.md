# Nova Robot Arm — Isaac Lab 圆柱体抓取放置训练计划（优化版）

> 任务目标：让 Nova 六自由度机械臂 + 夹爪在 Isaac Lab 中完成 **站立圆柱体抓取 → 搬运 → 放置到指定位置 → 松爪退出**，并为后续实机部署保留可迁移控制接口。

---

## 0. 这次优化的核心结论

相较于当前 plan，本版做 6 个关键调整：

1. **主线算法从“PPO-HER”调整为“IK-Rel + SAC/HER 主线，PPO 为保留基线”**
   - HER 在原始论文中是为 off-policy 方法提出的；
   - 对 pick-and-place 这类 goal-conditioned 稀疏奖励任务，`SAC + HER` / `TD3 + HER` 是更标准、更省工程量的组合；
   - 当前 PPO-HER 可保留为对照实验，但不建议作为唯一主线。

2. **动作空间从“直接 6 关节控制”改为“末端相对位姿 + 夹爪”**
   - 训练阶段优先使用 `EE delta xyz (+ yaw) + gripper`；
   - 底层使用 Differential IK / Operational Space Controller 转成关节命令；
   - 这样比直接学 6 维关节搜索更容易收敛。

3. **将任务显式拆成 4 个子阶段**
   - Reach（接近）
   - Grasp（夹取）
   - Lift & Transport（提起与搬运）
   - Place & Release（放置与松开）

4. **针对“圆柱体”重写目标表示**
   - 圆柱体轴对称，通常不需要优化绕自身轴的 yaw；
   - 任务更应关注：物体中心位置、是否保持直立、是否成功松爪后稳定留在目标区。

5. **加入强约束版奖励与成功判据**
   - 奖励不只看距离，还要看双指接触、提离桌面、带物接近目标、松爪后稳定停留；
   - 成功不只看最终 reward，要分解成 reach / grasp / lift / place 四类成功率。

6. **把 Sim2Real 提前纳入训练设计**
   - 训练阶段就加入物体质量、摩擦、夹爪摩擦、控制延迟、观测噪声随机化；
   - 部署时复用同一套“高层末端动作 → 低层控制器 → 电机执行”链路。

---

## 1. 为什么当前 plan 需要改

当前 plan 的方向是：
- 保留 joint-space 风格；
- 在 PPO 上硬加 HER；
- achieved_goal / desired_goal 只用 3D 位置；
- curriculum 主要靠“初始位置变近”。

这些改法能比纯 PPO 好，但仍有 4 个结构性问题：

### 1.1 PPO-HER 不是这类任务最自然的主线
HER 天然适合 off-policy replay。对操作任务来说，用 `SAC + HER` 或 `TD3 + HER` 的工程路径更直接：
- replay buffer 天然存在；
- hindsight relabel 更顺；
- 不需要在 on-policy rollout 里改很多逻辑。

### 1.2 直接 joint-space 搜索对 pick-and-place 太难
如果让策略直接输出 6 个关节动作，它要同时学会：
- 可达性；
- 末端绕障；
- 夹爪对齐；
- 接触时姿态控制；
- 抓起后运输。

对一个“抓圆柱放目标点”的任务来说，这个搜索空间没有必要这么大。

### 1.3 当前 HER 目标定义不够贴合抓放任务
如果 achieved_goal 只是“物体位置”，虽然适合放置，但不够表达：
- 是否已抓住；
- 是否已离桌；
- 是否竖直；
- 是否松爪后还稳定在目标区。

### 1.4 现在的课程学习只在缩短“初始距离”
这能解决 reach 的探索问题，但不能解决：
- 夹爪姿态对齐；
- 夹持稳定性；
- 松爪后的二次滑动；
- 目标区精确放置。

---

## 2. 优化后的总体技术路线

### 推荐主线（A）
**环境：Isaac Lab 自定义 pick-place 环境**  
**动作：IK-Rel 末端相对位姿控制 + gripper**  
**算法：SAC + HER**  
**奖励：阶段门控 dense reward + binary success**  
**迁移：domain randomization + 延迟/噪声建模**

### 对照线（B）
**环境与奖励相同**  
**算法：PPO（或 PPO-HER）**

### 保底线（C，可选）
如果 RL 两周内仍然卡在 grasp/lift：
- 采集 10–20 条 teleop 成功示教；
- 用 Isaac Lab Mimic 扩充数据；
- 先做 BC / robomimic 风格预训练；
- 再 RL 微调。

---

## 3. 圆柱体任务的正式定义

### 3.1 初始状态
- 圆柱体初始为“竖直放置在桌面”；
- 位置在可达工作区内随机；
- 半径、长度、质量、摩擦可做小范围随机；
- 目标放置点在桌面上随机采样（前期只采样桌面目标，不采样空中目标）。

### 3.2 成功条件（必须全部满足）
记：
- `p_obj` = 圆柱体中心位置
- `p_goal` = 目标位置
- `z_axis_obj` = 圆柱主轴方向
- `z_world = [0, 0, 1]`

Episode 成功定义为：
1. `||p_obj_xy - p_goal_xy|| < 0.03 m`
2. `|p_obj_z - p_goal_z| < 0.02 m`
3. `dot(z_axis_obj, z_world) > 0.95`（基本直立）
4. 夹爪已打开到释放阈值
5. 释放后连续 `N=10~20` 个控制周期内，物体仍停留在目标区

> 这里对圆柱体不要求绕自身轴旋转角（yaw），因为对称轴绕 z 的旋转通常不影响任务成功。

### 3.3 分阶段子目标
- **Stage A — Reach**：末端到圆柱预抓取位姿上方
- **Stage B — Grasp**：夹爪闭合并形成稳定夹持
- **Stage C — Lift / Transport**：物体抬离桌面并移动到目标上方
- **Stage D — Place / Release**：下放、松爪、退出，物体稳定留在目标区

---

## 4. 观测空间（重写）

不要再只做“世界坐标拼接”。建议用 **goal-conditioned + 相对量 + 任务状态量**。

### 4.1 policy observation（建议 32~40 维）

```python
obs = [
    # robot proprioception
    q[0:6],                    # 6, 关节角（归一化）
    dq[0:6],                   # 6, 关节速度（归一化）
    gripper_opening,           # 1

    # end-effector state
    ee_pos_base,               # 3, EE在基座坐标系位置
    ee_quat_base,              # 4, EE姿态（或改为6D rotation repr）
    ee_lin_vel_base,           # 3

    # object state
    obj_pos_base,              # 3
    obj_quat_base,             # 4
    obj_lin_vel_base,          # 3
    obj_ang_vel_base,          # 3

    # relative task features
    ee_to_obj,                 # 3 = obj_pos - ee_pos
    obj_to_goal,               # 3 = goal_pos - obj_pos
    ee_to_pregrasp,            # 3, 预抓取位姿相对向量

    # contact / phase features
    left_finger_contact,       # 1
    right_finger_contact,      # 1
    grasp_candidate,           # 1
    object_lifted,             # 1
]
```

### 4.2 achieved_goal / desired_goal（HER 用）

建议不要只用 3D 位置，而是使用：

```python
achieved_goal = [
    obj_pos_x, obj_pos_y, obj_pos_z,   # 3
    upright_score,                     # 1, dot(z_axis_obj, z_world)
]

desired_goal = [
    goal_pos_x, goal_pos_y, goal_pos_z,
    1.0,                               # 目标要求保持直立
]
```

如果你们后续确认只要求“放到点上，不在乎直立”，可以把第 4 维去掉，但对于圆柱体，直立约束通常非常重要。

### 4.3 不建议直接给 policy 的量
- 绝对世界坐标中的大范围无关量
- “oracle success flag” 直接作为输入
- 过多历史帧（先别加 RNN）

---

## 5. 动作空间（重点改）

### 5.1 推荐动作定义
训练时，策略输出：

```python
action = [dx, dy, dz, dyaw, g]
```

含义：
- `dx, dy, dz`：末端相对位移命令
- `dyaw`：末端绕竖直方向的小角度调整（圆柱通常够用）
- `g`：夹爪开/闭命令

如果你发现仅用 yaw 不够对齐，再升到 6D：
```python
[dx, dy, dz, droll, dpitch, dyaw, g]
```

### 5.2 底层控制
- Isaac Lab 里用 `DifferentialIKController` 或 OSC
- 生成期望 EE pose
- 再映射为关节命令发给机器人

### 5.3 为什么不把 joint velocity 作为训练主动作
joint velocity 对 sim2real 迁移通常更有利，但对学习难度不一定最低。建议分层：

- **训练主动作**：EE 相对动作（更好学）
- **部署执行层**：IK / OSC → joint target / joint velocity（更好落地）

这样既保留学习效率，也保留实机接口一致性。

---

## 6. 奖励函数（按“阶段门控 + 稳定放置”重写）

### 6.1 总体原则
- 任务主成功信号仍保留 binary success；
- dense reward 只做引导，不替代成功定义；
- 后一阶段奖励在前一阶段基本满足后再显著生效；
- 奖励项尽量对应真实物理事件，而不是抽象距离堆叠。

### 6.2 建议奖励项

记：
- `d_eo = ||ee_pos - pregrasp_pos||`
- `d_og = ||obj_pos - goal_pos||`
- `h = obj_pos_z - table_height`

```python
r = (
    0.6 * r_reach
  + 1.0 * r_alignment
  + 1.2 * r_grasp_contact
  + 1.5 * r_lift
  + 1.2 * r_transport
  + 1.5 * r_place
  + 1.0 * r_release_stable
  - 0.02 * r_action_l2
  - 0.01 * r_joint_vel_l2
  - 0.20 * r_table_collision
  - 0.05 * r_joint_limit
)
```

#### (1) Reach reward
```python
r_reach = 1 - tanh(5.0 * d_eo)
```
作用：引导末端靠近“预抓取位姿”，而不是直接去撞圆柱中心。

#### (2) Alignment reward
用于鼓励夹爪法向与圆柱侧面法向对齐，或鼓励夹爪两指中心与圆柱轴线横向对齐。

可简化为：
```python
r_alignment = 1 - tanh(8.0 * lateral_misalignment)
```

#### (3) Grasp contact reward
要求：
- 左指接触 = 1
- 右指接触 = 1
- 夹爪正在闭合或已闭合到抓取区间

```python
r_grasp_contact = 1.0 if both_fingers_contact else 0.0
```

#### (4) Lift reward
只有当 grasp_candidate=True 时才强激活：
```python
r_lift = grasp_gate * (1 - tanh(10.0 * max(h_target - h, 0.0)))
```
建议 `h_target = 0.08 ~ 0.12m`

#### (5) Transport reward
仅在 `object_lifted=True` 时启用：
```python
r_transport = lifted_gate * (1 - tanh(4.0 * d_og))
```

#### (6) Place reward
要求物体到目标区并逐步下降到放置高度：
```python
r_place = lifted_gate * in_goal_xy_bonus * (1 - tanh(10.0 * place_height_error))
```

#### (7) Release stable reward
松爪且物体在目标区内稳定：
```python
r_release_stable = 1.0 if released_and_stable else 0.0
```

#### (8) 终局成功奖励
```python
r_success = 5.0 if task_success else 0.0
```

> 终局奖励必须显著大于 shaping reward，否则容易学成“拿着物体悬在目标上方不放”。

### 6.3 抓取判定（不要只靠单一 contact）
建议：
```python
grasp_candidate = (
    left_finger_contact
    and right_finger_contact
    and gripper_opening < close_threshold
    and obj_height > table_height + 0.01
)
```

更稳妥可加：
- 物体相对夹爪的速度较小
- 物体未与桌面持续接触

---

## 7. HER 方案（重写为“真正适合抓放任务”的版本）

### 7.1 推荐：SAC + HER 主线
- replay buffer 原生支持 hindsight relabel；
- 适合 goal-conditioned 稀疏奖励；
- 比在 PPO rollout 中硬加 HER 更自然。

### 7.2 HER relabel 策略
使用 `future` 策略：
- 每条 episode 中，对每个 transition 采样 `k=4~8` 个未来 achieved goal；
- 重新计算 reward；
- 真实样本 : HER 样本 = `1 : 4` 起步。

### 7.3 HER goal 维度
推荐 relabel：
```python
achieved_goal = [obj_x, obj_y, obj_z, upright_score]
```
而不是末端位置。

原因：
- 你的真正任务目标是“物体被放到哪里”，不是“手到了哪里”。
- 如果 relabel EE 位置，容易学成“末端接近成功”，但物体没拿起来。

### 7.4 何时不要 relabel
以下 transition 不建议大量 relabel：
- 物体已掉落滚走、姿态混乱；
- 手与物体完全无关的早期随机晃动；
- 接近 episode 末端的明显无效碰撞段。

可以加入简单过滤：
- 只有当 `min_ee_obj_dist < d_thresh` 或 `object_moved=True` 时才允许 HER 样本进入高权重池。

---

## 8. 课程学习（从“缩短距离”升级为“分任务课程”）

### Curriculum 0：Reach-only 预热（可选，建议）
- 不要求抓取；
- 目标仅为到达预抓取位姿；
- 成功阈值放宽；
- 训练 100k~300k steps。

### Curriculum 1：Easy Grasp
- 圆柱体固定在桌面中央附近；
- 只随机 2D 位置的小范围偏移；
- 目标点固定在近处；
- 允许 30~40% episode 从预抓取附近开始。

### Curriculum 2：Lift & Short Transport
- 放开圆柱体位置采样范围；
- 目标点仍在近距离区域；
- 加入 lift / transport 奖励。

### Curriculum 3：Full Pick-and-Place
- 物体和目标点全范围采样；
- 加入 release-stable 成功判定；
- 恢复正常 HOME reset。

### Curriculum 4：Robustness
- 开启 domain randomization；
- 增加控制延迟、观测噪声、轻微初始姿态偏差；
- 加入失败恢复 reset。

> 重点：课程不只是“离得更近”，而是“少一个子技能 → 多一个子技能”。

---

## 9. 随机化与 Sim2Real

### 9.1 训练期必须随机化的量

#### 物体相关
- 半径：`±10%`
- 高度：`±10%`
- 质量：`0.5x ~ 1.5x`
- 静摩擦 / 动摩擦：随机
- 初始 xy 偏移：工作区随机
- 初始 yaw：`[0, 2π)`

#### 机械臂 / 夹爪相关
- 控制延迟：`0~2` step
- 关节观测噪声
- 末端位姿噪声
- 夹爪闭合实际到位偏差
- 指尖摩擦系数随机

#### 场景相关
- 桌面摩擦
- 桌面高度微偏差
- 目标点微扰动

### 9.2 部署前一致性检查
必须对齐：
- 控制频率 `dt`
- 夹爪开闭时间常数
- 最大关节速度
- 限位与软限位
- 夹爪接触几何与实物等效宽度

### 9.3 部署链路
建议统一成：

```text
Policy (EE delta pose + gripper)
    ↓
Differential IK / OSC
    ↓
Joint target or joint velocity target
    ↓
joint_gui_web.py / execute_move(...)
    ↓
Motor driver / MIT mode
```

而不是训练时一个控制接口、部署时再临时改另一套。

---

## 10. 训练配置建议

## 10.1 算法优先级

### 主线：SAC + HER
建议初始配置：
```python
buffer_size = 1_000_000
batch_size = 1024
learning_rate = 3e-4
gamma = 0.99
tau = 0.005
train_freq = 1
gradient_steps = 1
ent_coef = "auto"
her_k = 4
```

### 对照：PPO
如果保留 PPO，对照时必须满足：
```python
entropy_coef >= 0.01
num_steps_per_env = 64 ~ 128
clip_range = 0.2
learning_rate = 3e-4
```

但 PPO 不建议再承担“主线救火”角色。

### 10.2 并行环境数
- SAC/HER：`256 ~ 1024` env 起步
- PPO：`1024 ~ 4096` env 视显存而定

### 10.3 episode 长度
建议：
- 控制频率 20~30 Hz
- 单 episode 120~180 steps

不要太短，否则还没完成“抓-提-移-放”就截断。

---

## 11. 验证指标（重写）

不要只看 mean reward，建议分 5 级指标：

### 11.1 子任务成功率
- `reach_success_rate`
- `grasp_success_rate`
- `lift_success_rate`
- `place_success_rate`
- `release_stable_success_rate`

### 11.2 终局成功率
- 最终 task success rate（严格按第 3.2 节定义）

### 11.3 轨迹质量指标
- 平均 episode 时长
- 平均夹爪闭合次数
- 平均 table collision 次数
- 平均放置误差（cm）

### 11.4 稳健性评估
- 质量变化下成功率
- 摩擦变化下成功率
- 控制延迟变化下成功率

### 11.5 Sim2Real 前门槛
建议进入实机测试前至少达到：
- sim task success `> 85%`
- 3 组随机化配置下 success `> 70%`
- 放置位置误差 P90 `< 4 cm`

---

## 12. 代码结构调整建议

```text
/home/bsrl/hongsenpang/nova_training/
├── assets/
│   ├── arm.usd
│   ├── cylinder.usd                  # 新增：标准圆柱体资产
│   └── meshes/
├── envs/
│   ├── __init__.py
│   ├── nova_pick_place_env.py        # 重构：goal-conditioned env
│   ├── reward_terms.py               # 新增：奖励项拆分
│   ├── success_utils.py              # 新增：success / grasp 判断
│   ├── curriculum.py                 # 新增：课程逻辑
│   ├── randomization.py              # 新增：domain randomization
│   └── wrappers_goal_env.py          # 新增：适配 HER / SB3 / skrl
├── scripts/
│   ├── train_ppo.py                  # 原 PPO 保留为 baseline
│   ├── train_sac_her.py              # 主训练脚本（新增）
│   ├── eval_policy.py                # 统一评估脚本（新增）
│   ├── record_video.py
│   ├── eval_video.py
│   └── teleop_collect.py             # 可选：采集示教
├── configs/
│   ├── env/
│   │   └── nova_pick_place_cylinder.yaml
│   ├── train/
│   │   ├── sac_her.yaml
│   │   └── ppo_baseline.yaml
│   └── randomization/
│       └── sim2real.yaml
├── logs/
└── video/
```

---

## 13. 具体实施步骤（按周 / 按里程碑）

## Milestone 1：任务重构
- [ ] 替换 object 为标准圆柱体资产
- [ ] 定义严格 success 条件
- [ ] 完成 observation / achieved_goal / desired_goal 重构
- [ ] 切换动作为 IK-Rel EE delta + gripper
- [ ] 手动 teleop 验证：人可以稳定完成抓放

**验收**：键盘 / spacemouse 下 20 次中至少 15 次可成功抓放。

## Milestone 2：奖励与课程
- [ ] 实现分阶段 reward
- [ ] 实现 grasp_candidate / object_lifted 判定
- [ ] 实现 4 阶课程学习
- [ ] 跑随机动作 + scripted policy 做 reward sanity check

**验收**：
- reach 奖励单调；
- 抓起时 lift reward 明显跳升；
- 成功放置后 release_stable 触发。

## Milestone 3：SAC + HER 主训练
- [ ] 接入 SAC + HER
- [ ] future goal relabel, k=4 起步
- [ ] 256 env 冒烟测试
- [ ] 512 / 1024 env 正式训练

**验收**：
- 200k~500k steps 内出现稳定 grasp；
- 1M~2M steps 内出现稳定 place；
- final success > 60%。

## Milestone 4：对照实验
- [ ] 跑 PPO baseline
- [ ] 记录 PPO / PPO-HER / SAC-HER 三组曲线
- [ ] 对比 sample efficiency 和 success rate

**验收**：形成一页内部结论：哪条路线继续投入。

## Milestone 5：Sim2Real
- [ ] 打开 domain randomization
- [ ] 对齐控制频率与限位
- [ ] 实机低速推理测试
- [ ] 先做“抓起不放”
- [ ] 再做“抓起后搬运”
- [ ] 最后做完整“放置并松爪”

---

## 14. 风险点与预案

### 风险 1：圆柱体总是被推倒，难以形成稳定抓取
**处理：**
- 把预抓取位姿设为“上方偏侧”而不是正撞中心；
- 减小早期动作尺度；
- 前期增大 alignment reward；
- 先固定圆柱半径与高度，不要一开始就全随机。

### 风险 2：学会夹住但不会松爪放置
**处理：**
- 提高终局 success bonus；
- release_stable reward 必须显著；
- place reward 在“已释放且稳定”时高于“抓着悬停”。

### 风险 3：sim 里能抓，实机会滑落
**处理：**
- 训练期随机化接触参数；
- 实机先标定夹爪闭合宽度与力限；
- 必要时在策略外增加最小闭合保持逻辑。

### 风险 4：SAC + HER 与现有代码栈耦合麻烦
**处理：**
- 保留现有 PPO 脚本；
- 新建 `train_sac_her.py` 与独立 wrapper；
- 不在旧 `train.py` 上堆 patch。

### 风险 5：纯 RL 仍卡在 grasp
**处理：**
- 启动 Mimic / teleop 路线；
- 用少量成功示教做 warm-start。

---

## 15. 最终建议（结论版）

如果这就是一个**“六自由度机械臂抓站立圆柱并放到指定位置”**的单任务项目，那么最值得改的不是再继续细调 PPO，而是把 plan 改成下面这条线：

### 最推荐版本
- **动作**：`EE relative delta pose + gripper`
- **算法**：`SAC + HER`
- **观测**：goal-conditioned，相对坐标 + 接触/抓取状态
- **奖励**：reach → align → grasp → lift → transport → place → release stable
- **课程**：按技能阶段递进，不只按初始距离递进
- **随机化**：质量 / 摩擦 / 尺寸 / 延迟 / 噪声
- **部署**：统一走 IK / OSC → joint command → motor

### 保留项
- 你 current plan 中“goal-conditioned 观测”“relative features”“课程学习”“Sim2Real 意识”这些思路是对的，可以保留。

### 应该替换项
- 把“PPO-HER 作为唯一主线”替换掉；
- 把“joint velocity 直接作为训练动作”替换成“末端动作 + 低层控制器”；
- 把“achieved_goal = 单纯 3D 物体位置”升级为“位置 + uprightness”；
- 把“只有 mean reward / success rate”升级为分阶段成功率看板。

---

## 16. 参考依据（用于 plan 注释区）

### Papers
- Andrychowicz et al., **Hindsight Experience Replay**, 2017
- Peng et al., **Sim-to-Real Transfer of Robotic Control with Dynamics Randomization**, 2017
- Bertoni et al., **On the Role of the Action Space in Robot Manipulation Learning and Sim-to-Real Transfer**, 2023

### Official docs / projects
- **Gymnasium-Robotics Fetch PickAndPlace**（goal-conditioned 观测、HER 兼容接口、Cartesian gripper action）
- **Isaac Lab** manipulation environments（Franka Lift Cube、IK-Rel、Differential IK）
- **IsaacGymEnvs FrankaCubeStack**（staged reward / manipulation baseline）
- **robosuite PickPlace**（reaching / grasping / lifting / hovering staged rewards）
- **Isaac Lab Mimic**（少量示教 + 自动扩增）

