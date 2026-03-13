"""训练后推理录像（加载最新 checkpoint，录制30秒策略效果）

用法:
    export CONDA_PREFIX=/home/bsrl/miniconda3/envs/thunder2
    export ISAACLAB_PATH=/home/bsrl/hongsenpang/RLbased/IsaacLab
    bash /home/bsrl/IsaacLab/isaaclab.sh -p /home/bsrl/hongsenpang/nova_training/scripts/eval_video.py
    bash /home/bsrl/IsaacLab/isaaclab.sh -p .../eval_video.py --checkpoint /path/to/model_300.pt
"""
import argparse
import sys
import os
import glob

sys.path.insert(0, "/home/bsrl/hongsenpang/nova_training")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Nova 推理录像")
parser.add_argument("--checkpoint", type=str, default="",
                    help="checkpoint 路径，留空则自动选最新")
parser.add_argument("--video_dir", type=str,
                    default="/home/bsrl/hongsenpang/nova_training/video")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from envs.nova_pick_place_env import NovaPickPlaceEnv, NovaPickPlaceEnvCfg
from rsl_rl.modules import ActorCritic

NUM_STEPS   = 750   # 30秒 @ 25Hz
LOG_DIR     = "/home/bsrl/hongsenpang/nova_training/logs"
OBS_DIM     = 34
ACT_DIM     = 7

# ── 自动找最新 checkpoint ─────────────────────────────────────────────────────
def find_latest_checkpoint(log_dir: str) -> str:
    pattern = os.path.join(log_dir, "**", "model_*.pt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"在 {log_dir} 下未找到任何 checkpoint")
    # 按文件修改时间取最新
    latest = max(files, key=os.path.getmtime)
    return latest

ckpt_path = args.checkpoint if args.checkpoint else find_latest_checkpoint(LOG_DIR)
print(f"[INFO] 使用 checkpoint: {ckpt_path}")

# ── 加载策略网络 ──────────────────────────────────────────────────────────────
policy = ActorCritic(
    num_actor_obs=OBS_DIM,
    num_critic_obs=OBS_DIM,
    num_actions=ACT_DIM,
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    init_noise_std=1.0,
)
ckpt = torch.load(ckpt_path, map_location="cpu")
# RSL-RL checkpoint 结构: {"model_state_dict": {...}}
state_dict = ckpt.get("model_state_dict", ckpt)
policy.load_state_dict(state_dict)
policy.eval()
print("[INFO] 策略网络加载完成")

# ── 构建环境 ──────────────────────────────────────────────────────────────────
cfg = NovaPickPlaceEnvCfg()
cfg.scene.num_envs = 1
env = NovaPickPlaceEnv(cfg, render_mode="rgb_array")

os.makedirs(args.video_dir, exist_ok=True)
env = gym.wrappers.RecordVideo(
    env,
    args.video_dir,
    step_trigger=lambda step: step == 0,
    video_length=NUM_STEPS,
    disable_logger=True,
)

print(f"[INFO] 开始推理录制 {NUM_STEPS} 步（约30秒），输出: {args.video_dir}")

obs_dict, _ = env.reset()
obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
device = next(policy.parameters()).device

with torch.inference_mode():
    for i in range(NUM_STEPS):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        action = policy.act_inference(obs_tensor)          # (1, 7)
        action_np = action.squeeze(0).cpu().numpy()

        obs_dict, rew, term, trunc, info = env.step(action_np)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict

        if i % 100 == 0:
            r = rew.item() if hasattr(rew, "item") else float(rew)
            print(f"  step {i}/{NUM_STEPS}, rew={r:.3f}")

        # episode 结束后自动 reset（gymnasium wrapper 会处理，此处仅打印）
        if term or trunc:
            print(f"  [episode done at step {i}]")

env.close()
simulation_app.close()
print(f"[INFO] 完成，视频保存在 {args.video_dir}")
