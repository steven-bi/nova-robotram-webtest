"""录制仿真视频（验证场景是否正确）

用法:
    export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2
    export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab
    bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/record_video.py
"""
import sys, os
sys.path.insert(0, "/home/bsrl/hongsenpang/nova_training")

from isaaclab.app import AppLauncher

# 必须 enable_cameras 才能 render_mode=rgb_array
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from envs.nova_pick_place_env import NovaPickPlaceEnv, NovaPickPlaceEnvCfg

VIDEO_DIR = "/home/bsrl/hongsenpang/nova_training/video"
NUM_STEPS = 750  # 30秒 @ 25Hz policy

# ── 环境（render_mode=rgb_array 触发相机渲染）───────────────────────────────
cfg = NovaPickPlaceEnvCfg()
cfg.scene.num_envs = 1

env = NovaPickPlaceEnv(cfg, render_mode="rgb_array")

# ── 视频录制 wrap ─────────────────────────────────────────────────────────────
os.makedirs(VIDEO_DIR, exist_ok=True)
env = gym.wrappers.RecordVideo(
    env,
    VIDEO_DIR,
    step_trigger=lambda step: step == 0,   # 从第0步开始录
    video_length=NUM_STEPS,
    disable_logger=True,
)

print(f"[INFO] 开始录制 {NUM_STEPS} 步，输出目录: {VIDEO_DIR}")

obs, _ = env.reset()

with torch.inference_mode():
    for i in range(NUM_STEPS):
        if i < 100:
            action = torch.zeros(1, 7)            # 前100步静止观察场景
        else:
            action = torch.randn(1, 7) * 0.3      # 之后随机动作
            action[:, 6] = 1.0 if (i // 60) % 2 == 0 else -1.0

        obs, rew, term, trunc, info = env.step(action)
        if i % 100 == 0:
            print(f"  step {i}/{NUM_STEPS}, rew={rew.item() if hasattr(rew, 'item') else rew:.3f}")

env.close()
simulation_app.close()
print(f"[INFO] 完成，视频保存在 {VIDEO_DIR}")
