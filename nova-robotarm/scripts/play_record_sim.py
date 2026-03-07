"""Play trained policy and record video using TiledCamera sensor."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--video_length", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import imageio
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# Import with camera added to scene
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from arm_grasp.envs.lift_cube_cfg import ArmLiftCubeEnvCfg_PLAY, ArmLiftSceneCfg
from arm_grasp.agents import ArmLiftCubePPORunnerCfg


# Create a scene config with camera
@configclass
class SceneWithCameraCfg(ArmLiftSceneCfg):
    """Scene config that adds a camera for recording."""
    tiled_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.7, -0.7, 1.1),
            rot=(1.0, 0.0, 0.0, 0.0),  # placeholder, will be set in code
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
class PlayWithCameraCfg(ArmLiftCubeEnvCfg_PLAY):
    scene: SceneWithCameraCfg = SceneWithCameraCfg(num_envs=1, env_spacing=2.5)


def main():
    cfg = PlayWithCameraCfg()
    env = ManagerBasedRLEnv(cfg=cfg)

    # Load trained policy
    agent_cfg = ArmLiftCubePPORunnerCfg()
    agent_cfg.resume = True

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs", "rsl_rl", agent_cfg.experiment_name
    )

    env_wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device="cuda:0")

    resume_path = os.path.join(log_dir, "model_4500.pt")
    print("Loading policy from:", resume_path)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Set camera look-at using USD
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/Camera")
    if cam_prim.IsValid():
        xformable = UsdGeom.Xformable(cam_prim)
        xformable.ClearXformOpOrder()
        # Eye position and look-at target
        eye = Gf.Vec3d(0.7, -0.7, 1.1)
        target = Gf.Vec3d(0.15, 0.0, 0.85)
        up = Gf.Vec3d(0, 0, 1)
        # Compute look-at matrix
        fwd = (target - eye).GetNormalized()
        right = (fwd ^ up).GetNormalized()
        new_up = (right ^ fwd).GetNormalized()
        # Camera in USD looks along -Z, so negate forward
        mat = Gf.Matrix4d()
        mat.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
        mat.SetRow(1, Gf.Vec4d(new_up[0], new_up[1], new_up[2], 0))
        mat.SetRow(2, Gf.Vec4d(-fwd[0], -fwd[1], -fwd[2], 0))
        mat.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
        xformable.AddTransformOp().Set(mat)
        print("Camera set: eye=%s target=%s" % (eye, target))
    else:
        print("WARNING: Camera prim not found, trying default")

    # Get camera sensor
    camera = env.scene["tiled_camera"]
    print("Camera sensor:", camera)

    print("Recording %d steps..." % args.video_length)

    obs, _ = env.reset()
    obs_dict = env_wrapped.get_observations()

    frames = []
    for step in range(args.video_length):
        with torch.no_grad():
            actions = policy(obs_dict)

        obs_dict, rew, dones, extras = env_wrapped.step(actions)

        # Get camera image
        try:
            img_data = camera.data.output["rgb"]
            if img_data is not None:
                # img_data shape: (num_envs, H, W, C) on GPU
                frame = img_data[0].cpu().numpy()
                if frame.dtype == np.float32:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                frames.append(frame[:, :, :3].copy())
        except Exception as e:
            if step == 0:
                print("Camera capture error:", e)
                # Try alternative access
                try:
                    print("Camera data keys:", dir(camera.data))
                    if hasattr(camera.data, 'output'):
                        print("Output keys:", camera.data.output.keys() if hasattr(camera.data.output, 'keys') else type(camera.data.output))
                except:
                    pass

        if step % 50 == 0:
            ee_pos = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy()
            obj_pos = env.scene["object"].data.root_pos_w[0].cpu().numpy()
            dist = np.linalg.norm(ee_pos - obj_pos)
            print("Step %d: dist=%.3fm rew=%.3f obj_z=%.3f frames=%d" % (
                step, dist, rew[0].item(), obj_pos[2], len(frames)))

    # Save video
    if len(frames) > 10:
        video_path = "/root/autodl-tmp/arm_grasp/policy_video.mp4"
        imageio.mimwrite(video_path, frames, fps=30, quality=8)
        print("Video saved: %s (%d frames, %.1fs)" % (video_path, len(frames), len(frames)/30.0))
    else:
        print("Only got %d frames" % len(frames))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
