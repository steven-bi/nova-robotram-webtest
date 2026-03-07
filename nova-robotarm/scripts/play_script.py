"""Play trained policy and record trajectory data."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import json
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from arm_grasp.envs.lift_cube_cfg import ArmLiftCubeEnvCfg_PLAY
from arm_grasp.agents import ArmLiftCubePPORunnerCfg


def main():
    cfg = ArmLiftCubeEnvCfg_PLAY()
    cfg.scene.num_envs = 4
    env = ManagerBasedRLEnv(cfg=cfg)

    agent_cfg = ArmLiftCubePPORunnerCfg()
    agent_cfg.resume = True

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rsl_rl", agent_cfg.experiment_name
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    # Load latest policy
    resume_path = os.path.join(log_dir, "model_4500.pt")
    print("Loading policy from:", resume_path)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Access scene objects
    robot = env.scene["robot"]
    ee_frame = env.scene["ee_frame"]
    obj = env.scene["object"]

    print("")
    print("=== Running Trained Policy ===")
    print("")

    obs, _ = env.reset()
    obs_dict = env_wrapped.get_observations()

    trajectory = []
    total_reward = 0.0

    for step in range(400):
        with torch.no_grad():
            actions = policy(obs_dict)

        obs_dict, rew, dones, extras = env_wrapped.step(actions)
        total_reward += rew.mean().item()

        # Collect state
        ee_pos = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
        obj_pos = obj.data.root_pos_w[0].cpu().numpy()
        jp = robot.data.joint_pos[0].cpu().numpy()

        frame = {
            "step": step,
            "ee_x": float(ee_pos[0]), "ee_y": float(ee_pos[1]), "ee_z": float(ee_pos[2]),
            "obj_x": float(obj_pos[0]), "obj_y": float(obj_pos[1]), "obj_z": float(obj_pos[2]),
            "joints": [float(j) for j in jp[:8]],
            "reward": float(rew[0].item()),
            "distance": float(np.linalg.norm(ee_pos - obj_pos)),
        }
        trajectory.append(frame)

        if step % 20 == 0:
            dist = np.linalg.norm(ee_pos - obj_pos)
            msg = "Step %3d: EE=[%.3f,%.3f,%.3f] Obj=[%.3f,%.3f,%.3f] dist=%.4fm rew=%.3f" % (
                step, ee_pos[0], ee_pos[1], ee_pos[2],
                obj_pos[0], obj_pos[1], obj_pos[2],
                dist, rew[0].item()
            )
            print(msg)

    print("")
    print("Total reward: %.2f" % total_reward)
    print("Mean step reward: %.4f" % (total_reward / 400))

    # Save trajectory
    out_path = "/root/autodl-tmp/arm_grasp/trajectory_data.json"
    with open(out_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print("Trajectory saved to", out_path)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
