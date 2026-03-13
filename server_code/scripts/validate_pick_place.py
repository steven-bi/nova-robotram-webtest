"""Validate pick-and-place v2 environment loads and runs."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from arm_grasp.envs.pick_place_cfg import ArmPickPlaceEnvCfg

cfg = ArmPickPlaceEnvCfg()
cfg.scene.num_envs = 4
env = ManagerBasedRLEnv(cfg=cfg)

obs, _ = env.reset()

print("=== Pick-and-Place v2 Environment Validation ===")
print(f"Obs shape: {obs['policy'].shape}")
print(f"Action space: {env.action_space}")

robot = env.scene["robot"]
ee_frame = env.scene["ee_frame"]
obj = env.scene["object"]
print(f"Robot pos: {robot.data.root_pos_w[0].cpu().numpy()}")
print(f"EE pos: {ee_frame.data.target_pos_w[0, 0].cpu().numpy()}")
print(f"Cup pos: {obj.data.root_pos_w[0].cpu().numpy()}")
print(f"Cup quat: {obj.data.root_quat_w[0].cpu().numpy()}")

# Run 5 steps with random actions
for step in range(5):
    action = torch.zeros(4, env.action_space.shape[-1], device=env.device)
    obs, rew, term, trunc, info = env.step(action)
    ee_pos = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
    cup_pos = obj.data.root_pos_w[0].cpu().numpy()
    print(f"Step {step}: EE={ee_pos}, Cup={cup_pos}, rew={rew[0].item():.3f}")

print("\n=== VALIDATION PASSED ===")
env.close()
simulation_app.close()
