"""PPO-HER 训练脚本

改进点（相比 train.py）：
- 使用 NovaHEREnv（相对观测 + HER reset + 课程学习）
- 超参数针对 HER 调优：entropy_coef=0.001, lr=3e-4, num_steps_per_env=64
- 每 500 轮自动推进课程阶段
- 每 500 轮在 log 中打印 success_rate 统计

用法：
    export CUDA_VISIBLE_DEVICES=4,5,6
    export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2
    export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab
    bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/train_her.py --headless
"""
import argparse
import sys
import os

sys.path.insert(0, "/home/bsrl/hongsenpang/nova_training")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Nova 机械臂 PPO-HER 训练")
parser.add_argument("--num_envs",       type=int,   default=1024)
parser.add_argument("--max_iterations", type=int,   default=5000)
parser.add_argument("--log_dir",        type=str,   default="/home/bsrl/hongsenpang/nova_training/logs_her")
parser.add_argument("--her_prob",       type=float, default=0.5,  help="HER 目标使用概率")
parser.add_argument("--curriculum",     action="store_true",       help="启用课程学习")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from envs.nova_her_env import NovaHEREnv, NovaHEREnvCfg

# ── 环境配置 ────────────────────────────────────────────────────────────────
env_cfg = NovaHEREnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.her_prob       = args.her_prob

env = NovaHEREnv(env_cfg)
env = RslRlVecEnvWrapper(env)

# 课程阶段更新辅助（每 500 轮推进一次，共 4 阶段）
CURRICULUM_INTERVAL = 500
MAX_CURRICULUM_STAGE = 3

def update_curriculum(iteration: int):
    """根据训练轮数更新物体生成范围。"""
    if not args.curriculum:
        env.unwrapped.cfg.curriculum_stage = MAX_CURRICULUM_STAGE  # 直接用最终阶段
        return
    stage = min(iteration // CURRICULUM_INTERVAL, MAX_CURRICULUM_STAGE)
    if stage != env.unwrapped.cfg.curriculum_stage:
        env.unwrapped.cfg.curriculum_stage = stage
        print(f"\n[Curriculum] 推进到阶段 {stage} (iter {iteration})\n")

# ── 训练配置 ────────────────────────────────────────────────────────────────
train_cfg = RslRlOnPolicyRunnerCfg(
    seed=42,
    device=str(env.device),
    num_steps_per_env=64,        # 更长的 rollout，优势估计更准
    max_iterations=args.max_iterations,
    save_interval=500,
    experiment_name="nova_her",
    run_name="",
    resume=False,
    load_run=".*",
    load_checkpoint="model_.*.pt",
    logger="tensorboard",
    neptune_project="isaaclab",
    wandb_project="isaaclab",
    obs_groups={"policy": ["policy"]},

    policy=RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,      # 降低熵系数，防止 noise_std 爆炸
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)

# ── 自定义训练循环（支持课程更新）───────────────────────────────────────────
os.makedirs(args.log_dir, exist_ok=True)

from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, train_cfg.to_dict(), log_dir=args.log_dir, device=str(env.device))

print(f"\n开始 PPO-HER 训练")
print(f"  并行环境: {args.num_envs}")
print(f"  最大轮数: {args.max_iterations}")
print(f"  HER 概率: {args.her_prob}")
print(f"  课程学习: {'启用' if args.curriculum else '禁用（直接最终阶段）'}")
print(f"  日志目录: {args.log_dir}\n")

# 初始化课程阶段
update_curriculum(0)

# 分段训练，每 CURRICULUM_INTERVAL 轮更新一次课程
trained = 0
chunk = CURRICULUM_INTERVAL
while trained < args.max_iterations:
    this_chunk = min(chunk, args.max_iterations - trained)
    runner.learn(num_learning_iterations=this_chunk, init_at_random_ep_len=False)
    trained += this_chunk
    update_curriculum(trained)

env.close()
simulation_app.close()
print(f"\n[完成] 训练结束，checkpoint 保存在 {args.log_dir}")
