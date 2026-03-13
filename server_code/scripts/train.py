"""Train PPO policy for arm lift-cube task.

Usage:
    cd /root/autodl-tmp/arm_grasp
    /root/autodl-tmp/conda_envs/thunder2/bin/python scripts/train.py --num_envs 2048 --headless
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train arm lift-cube policy")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from arm_grasp.envs.lift_cube_cfg import ArmLiftCubeEnvCfg
from arm_grasp.agents import ArmLiftCubePPORunnerCfg


def main():
    cfg = ArmLiftCubeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed

    env = ManagerBasedRLEnv(cfg=cfg)

    agent_cfg = ArmLiftCubePPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rsl_rl", agent_cfg.experiment_name
    )
    os.makedirs(log_dir, exist_ok=True)

    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
