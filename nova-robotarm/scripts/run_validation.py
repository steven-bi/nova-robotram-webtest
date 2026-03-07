"""Run validation test on server."""
import paramiko

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.westd.seetacloud.com', port=14918, username='root', password='UvUnT2x1jsaa', timeout=15)

sftp = client.open_sftp()
with sftp.file('/root/autodl-tmp/arm_grasp/scripts/validate.py', 'w') as f:
    f.write('''"""Quick validation: create env, step a few times, print info."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

from arm_grasp.envs.lift_cube_cfg import ArmLiftCubeEnvCfg

def main():
    print("\\n=== Creating environment with 4 envs ===")
    cfg = ArmLiftCubeEnvCfg()
    cfg.scene.num_envs = 4
    env = ManagerBasedRLEnv(cfg=cfg)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Num envs: {env.num_envs}")

    obs, info = env.reset()
    print(f"\\nInitial obs shape: {obs['policy'].shape}")

    for i in range(5):
        action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward.mean().item():.4f}")

    print("\\n=== Validation PASSED! Environment works correctly. ===")
    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
''')
sftp.close()

print("Running validation on server...")
cmd = (
    'cd /root/autodl-tmp/arm_grasp && '
    'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
    '/root/autodl-tmp/conda_envs/thunder2/bin/python scripts/validate.py --headless 2>&1'
)
stdin, stdout, stderr = client.exec_command(cmd, timeout=300)
output = stdout.read().decode()
print(output[-4000:])
err = stderr.read().decode()
if err:
    print("STDERR (last 2000):", err[-2000:])

client.close()
