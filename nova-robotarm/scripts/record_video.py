"""Record video of arm_pick_place_v8 model_4999.pt using TiledCamera."""
import paramiko
import time
import os

SSH_HOST = 'connect.westd.seetacloud.com'
SSH_PORT = 14918
SSH_USER = 'root'
SSH_PASS = 'UvUnT2x1jsaa'
BASE = '/root/autodl-tmp/arm_grasp'
EXPERIMENT = 'arm_pick_place_v8_8'
MODEL = 'model_500.pt'
VIDEO_REMOTE = f'{BASE}/pick_place_v8_8.mp4'
VIDEO_LOCAL = 'pick_place_v8_8.mp4'

PLAY_SCRIPT = r'''"""Play and record pick-and-place v8 policy with TiledCamera."""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, "/root/autodl-tmp/arm_grasp")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import imageio
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils import configclass

from arm_grasp.envs.pick_place_cfg import ArmPickPlaceEnvCfg_PLAY, PickPlaceSceneCfg
from arm_grasp.agents import ArmPickPlacePPORunnerCfg

BASE = "/root/autodl-tmp/arm_grasp"
EXPERIMENT = "arm_pick_place_v8_8"
MODEL = "model_500.pt"


@configclass
class RecordSceneCfg(PickPlaceSceneCfg):
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.7, -0.7, 1.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=800,
        height=600,
    )


@configclass
class RecordEnvCfg(ArmPickPlaceEnvCfg_PLAY):
    scene: RecordSceneCfg = RecordSceneCfg(num_envs=1, env_spacing=3.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1  # Override PLAY default of 50


def set_camera_lookat(eye, target):
    try:
        import omni.usd
        from pxr import UsdGeom, Gf
        stage = omni.usd.get_context().get_stage()
        cam_prim = stage.GetPrimAtPath("/World/envs/env_0/Camera")
        if not cam_prim.IsValid():
            print("Camera prim not found")
            return
        xformable = UsdGeom.Xformable(cam_prim)
        xformable.ClearXformOpOrder()
        eye_g, tgt_g, up_g = Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)
        fwd = (tgt_g - eye_g).GetNormalized()
        right = (fwd ^ up_g).GetNormalized()
        new_up = (right ^ fwd).GetNormalized()
        mat = Gf.Matrix4d()
        mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
        mat.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
        mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
        mat.SetRow(3, Gf.Vec4d(eye_g[0], eye_g[1], eye_g[2], 1))
        xformable.AddTransformOp().Set(mat)
        print(f"Camera look-at set OK")
    except Exception as e:
        print(f"Camera look-at failed: {e}")


def main():
    cfg = RecordEnvCfg()
    env = ManagerBasedRLEnv(cfg=cfg)

    set_camera_lookat(eye=(0.7, -0.7, 1.1), target=(0.15, 0.0, 0.82))

    # Wrap env - policy expects flat tensor from RslRlVecEnvWrapper
    env_wrapped = RslRlVecEnvWrapper(env)

    agent_cfg = ArmPickPlacePPORunnerCfg()
    log_dir = os.path.join(BASE, "logs", "rsl_rl", EXPERIMENT)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")
    model_path = os.path.join(log_dir, MODEL)
    runner.load(model_path)
    policy = runner.get_inference_policy(device="cuda:0")
    print(f"Loaded: {model_path}")

    camera = env.scene["tiled_camera"]

    frames = []
    NUM_EPISODES = 3
    STEPS_PER_EP = 400

    for ep in range(NUM_EPISODES):
        # env_wrapped.reset() returns flat obs tensor
        obs, _ = env_wrapped.reset()
        print(f"\nEpisode {ep+1}/{NUM_EPISODES}")
        for step in range(STEPS_PER_EP):
            with torch.no_grad():
                actions = policy(obs)  # obs is flat tensor from wrapped env
            obs, rew, dones, extras = env_wrapped.step(actions)

            img = camera.data.output["rgb"]
            if img is not None and img.shape[0] > 0:
                frame = img[0].cpu().numpy()
                if frame.dtype != np.uint8:
                    frame = (frame.clip(0, 1) * 255).astype(np.uint8)
                frames.append(frame[:, :, :3].copy())

            if step % 100 == 0:
                obj = env.scene["object"]
                h = obj.data.root_pos_w[0, 2].item()
                r = rew[0].item()
                print(f"  step={step:3d}  obj_z={h:.3f}  rew={r:.3f}")

    print(f"\nCaptured {len(frames)} frames")
    out = "/root/autodl-tmp/arm_grasp/pick_place_v8_8.mp4"
    imageio.mimwrite(out, frames, fps=30, quality=8)
    size = os.path.getsize(out)
    print(f"Video saved: {out} ({size//1024} KB)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
'''


def run_cmd(client, cmd, timeout=30):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main():
    print("=" * 60)
    print("  Recording arm_pick_place_v8 (model_4999.pt)")
    print("=" * 60)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SSH_HOST, port=SSH_PORT, username=SSH_USER, password=SSH_PASS, timeout=15)
    print("Connected!")

    # Kill leftover processes
    client.exec_command('pkill -9 -f "play_record" 2>/dev/null', timeout=5)
    time.sleep(2)

    # Upload play script
    sftp = client.open_sftp()
    with sftp.file(f'{BASE}/scripts/play_record.py', 'w') as f:
        f.write(PLAY_SCRIPT)
    sftp.close()
    print("Uploaded play_record.py")

    # Launch via nohup — fire-and-forget (don't read stdout to avoid hang)
    LOG = f'{BASE}/record_video.log'
    nohup_cmd = (
        f'cd {BASE} && '
        'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && '
        f'nohup /root/autodl-tmp/conda_envs/thunder2/bin/python -u '
        f'scripts/play_record.py --headless --enable_cameras '
        f'> {LOG} 2>&1 & echo LAUNCHED'
    )
    # Use short timeout, don't block on stdout.read()
    stdin2, stdout2, stderr2 = client.exec_command(nohup_cmd, timeout=10)
    try:
        launch_out = stdout2.read(64).decode()  # Only read "LAUNCHED\n"
    except Exception:
        launch_out = "?"
    print(f"Recording launched: {launch_out.strip()}, log: {LOG}")

    # Poll until done (check for "Video saved" in log)
    print("Waiting for recording to finish (check every 30s)...")
    for attempt in range(20):
        time.sleep(30)
        _, log_tail, _ = run_cmd(client, f'tail -5 {LOG}', timeout=10)
        print(f"[{(attempt+1)*30}s] {log_tail.strip()}")

        # Check if done
        if 'Video saved' in log_tail or 'Error' in log_tail or 'Traceback' in log_tail:
            break

        # Check if process still running
        _, ps_out, _ = run_cmd(client, 'ps aux | grep play_record | grep -v grep', timeout=5)
        if not ps_out.strip():
            print("Process exited, getting full log...")
            _, full_log, _ = run_cmd(client, f'tail -50 {LOG}', timeout=10)
            print(full_log)
            break

    # Get final log
    _, full_log, _ = run_cmd(client, f'tail -30 {LOG}', timeout=10)
    print("\n--- Final log ---")
    print(full_log)

    # Check video exists
    _, ls_out, _ = run_cmd(client, f'ls -la {VIDEO_REMOTE} 2>/dev/null', timeout=10)
    if not ls_out.strip():
        print(f"Video not found: {VIDEO_REMOTE}")
        client.close()
        return

    print(f"\nVideo found: {ls_out.strip()}")

    # Download video
    print("Downloading video...")
    sftp = client.open_sftp()
    sftp.get(VIDEO_REMOTE, VIDEO_LOCAL)
    sftp.close()
    size = os.path.getsize(VIDEO_LOCAL)
    print(f"Downloaded: {VIDEO_LOCAL} ({size//1024} KB)")
    print(f"Full path: {os.path.abspath(VIDEO_LOCAL)}")

    client.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
