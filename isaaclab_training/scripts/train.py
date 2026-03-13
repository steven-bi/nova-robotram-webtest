"""训练入口脚本

用法:
    export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2
    export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab
    bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/train.py --headless
    bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/train.py --headless --num_envs 64
"""
import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, "/home/bsrl/hongsenpang/nova_training")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Nova 机械臂 Pick-and-Place 训练")
parser.add_argument("--num_envs", type=int, default=4096, help="并行环境数量")
parser.add_argument("--max_iterations", type=int, default=3000, help="最大训练轮数")
parser.add_argument("--log_dir", type=str,
                    default="/home/bsrl/hongsenpang/nova_training/logs",
                    help="日志保存目录")
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

from envs.nova_pick_place_env import NovaPickPlaceEnv, NovaPickPlaceEnvCfg

# ── 环境配置 ────────────────────────────────────────────────────────────────
env_cfg = NovaPickPlaceEnvCfg()
env_cfg.scene.num_envs = args.num_envs

env = NovaPickPlaceEnv(env_cfg)
env = RslRlVecEnvWrapper(env)

# ── 训练配置 (RSL-RL PPO) ───────────────────────────────────────────────────
train_cfg = RslRlOnPolicyRunnerCfg(
    seed=42,
    device=str(env.device),
    num_steps_per_env=48,   # 24→48，减少优势估计噪声
    max_iterations=args.max_iterations,
    save_interval=200,
    experiment_name="nova_pick_place",
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
        entropy_coef=0.005,     # 回退到 0.005，避免熵梯度主导 reward 信号
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,   # 1e-3→3e-4，更稳定
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    ),
)

# ── 开始训练 ────────────────────────────────────────────────────────────────
os.makedirs(args.log_dir, exist_ok=True)

from rsl_rl.runners import OnPolicyRunner
runner = OnPolicyRunner(env, train_cfg.to_dict(), log_dir=args.log_dir, device=str(env.device))

print(f"\n开始训练: {args.num_envs} 并行环境, 最大 {args.max_iterations} 轮")
print(f"日志目录: {args.log_dir}\n")

runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)

env.close()
simulation_app.close()
