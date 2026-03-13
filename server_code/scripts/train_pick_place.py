"""Train pick-and-place v2 policy with RSL-RL PPO."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from arm_grasp.envs.pick_place_cfg import ArmPickPlaceEnvCfg
from arm_grasp.agents import ArmPickPlacePPORunnerCfg


def main():
    cfg = ArmPickPlaceEnvCfg()
    if args.num_envs is not None:
        cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRLEnv(cfg=cfg)

    agent_cfg = ArmPickPlacePPORunnerCfg()
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rsl_rl", agent_cfg.experiment_name,
    )
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
