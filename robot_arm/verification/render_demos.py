"""
MuJoCo visualization demo for the 6-DOF robot arm.

Generates high-quality videos with proper lighting, ground plane, and colors:
1. Robot overview - rotating camera around multiple poses
2. Joint-by-joint demo - each joint sweeps its range
3. Circle trajectory tracking with EE trail
4. Pick-and-place trajectory
5. Workspace exploration
6. Multi-view circle

Usage:
    python -m robot_arm.verification.render_demos
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_arm.kinematics.kinematics import (
    forward_kinematics,
    inverse_kinematics,
    NUM_JOINTS,
)
from robot_arm.verification.mujoco_loader import (
    URDF_PATH,
    MESHES_DIR,
    MESH_FILES,
    set_joint_angles,
    get_ee_pose,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
VIDEO_W, VIDEO_H = 960, 720


def load_enhanced_model():
    """
    Load URDF, convert to MJCF, then add lighting, floor, skybox, and colors
    for better-looking renders.
    """
    # 1) Load the URDF as before
    with open(URDF_PATH, "r", encoding="utf-8") as f:
        urdf_str = f.read()
    urdf_fixed = urdf_str.replace("package://arm/meshes/", "")

    mesh_assets = {}
    for name in MESH_FILES:
        with open(os.path.join(MESHES_DIR, name), "rb") as f:
            mesh_assets[name] = f.read()

    # 2) Load into MuJoCo and save as MJCF XML
    tmp_model = mujoco.MjModel.from_xml_string(urdf_fixed, mesh_assets)
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "_arm_temp.xml")
    mujoco.mj_saveLastXML(tmp_path, tmp_model)
    with open(tmp_path, "r", encoding="utf-8") as f:
        mjcf_str = f.read()

    # 3) Parse XML and enhance it
    root = ET.fromstring(mjcf_str)

    # -- Add/replace <visual> section for better rendering --
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    # Global quality
    g = visual.find("global")
    if g is None:
        g = ET.SubElement(visual, "global")
    g.set("offwidth", str(VIDEO_W))
    g.set("offheight", str(VIDEO_H))
    # Quality settings
    quality = visual.find("quality")
    if quality is None:
        quality = ET.SubElement(visual, "quality")
    quality.set("shadowsize", "4096")
    quality.set("offsamples", "4")
    # Headlight
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("diffuse", "0.6 0.6 0.6")
    headlight.set("ambient", "0.3 0.3 0.3")
    headlight.set("specular", "0.2 0.2 0.2")

    # -- Add <asset> with textures and materials --
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    # Floor texture
    ET.SubElement(asset, "texture", {
        "name": "grid_tex", "type": "2d", "builtin": "checker",
        "rgb1": "0.85 0.85 0.85", "rgb2": "0.65 0.65 0.65",
        "width": "512", "height": "512",
    })
    ET.SubElement(asset, "material", {
        "name": "grid_mat", "texture": "grid_tex",
        "texrepeat": "8 8", "reflectance": "0.15",
    })
    # Skybox gradient
    ET.SubElement(asset, "texture", {
        "name": "sky_tex", "type": "skybox", "builtin": "gradient",
        "rgb1": "0.4 0.5 0.7", "rgb2": "0.9 0.9 1.0",
        "width": "512", "height": "3072",
    })
    # Robot link materials (colored)
    link_colors = {
        "base_mat":   "0.25 0.25 0.30",
        "link1_mat":  "0.20 0.45 0.70",
        "link2_mat":  "0.20 0.45 0.70",
        "link3_mat":  "0.85 0.55 0.15",
        "link4_mat":  "0.85 0.55 0.15",
        "link5_mat":  "0.20 0.65 0.40",
        "link6_mat":  "0.20 0.65 0.40",
        "grip_mat":   "0.35 0.35 0.35",
    }
    for mat_name, rgb in link_colors.items():
        ET.SubElement(asset, "material", {
            "name": mat_name, "rgba": rgb + " 1",
            "specular": "0.6", "shininess": "0.4", "reflectance": "0.1",
        })

    # -- Add floor and lights to worldbody --
    worldbody = root.find("worldbody")

    # Floor
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane",
        "pos": "0 0 0", "size": "2 2 0.1",
        "material": "grid_mat", "conaffinity": "1", "condim": "3",
    })
    # Key light (directional)
    ET.SubElement(worldbody, "light", {
        "name": "key_light", "pos": "0.5 -1.0 1.5",
        "dir": "-0.2 0.5 -0.7", "diffuse": "0.8 0.8 0.8",
        "specular": "0.3 0.3 0.3", "directional": "true",
        "castshadow": "true",
    })
    # Fill light (softer, from opposite side)
    ET.SubElement(worldbody, "light", {
        "name": "fill_light", "pos": "-0.5 1.0 1.0",
        "dir": "0.2 -0.5 -0.5", "diffuse": "0.4 0.4 0.5",
        "specular": "0.1 0.1 0.1", "directional": "true",
        "castshadow": "false",
    })

    # -- Apply colored materials to robot geoms --
    body_material_map = {
        "0": "base_mat",
        "1": "link1_mat",
        "2": "link2_mat",
        "3": "link3_mat",
        "4": "link4_mat",
        "5": "link5_mat",
        "6": "link6_mat",
        "7-1": "grip_mat",
        "7-2": "grip_mat",
    }

    for body in worldbody.iter("body"):
        body_name = body.get("name", "")
        mat_name = body_material_map.get(body_name)
        if mat_name:
            for geom in body.findall("geom"):
                geom.set("material", mat_name)

    # Also set base link geoms (direct children of worldbody)
    for geom in worldbody.findall("geom"):
        gname = geom.get("name", "")
        if gname and "floor" not in gname:
            geom.set("material", "base_mat")

    # 4) Convert back to string and load
    enhanced_xml = ET.tostring(root, encoding="unicode")

    # Collect mesh assets from the saved model's references
    # The MJCF references mesh files by the names in <asset><mesh>
    model = mujoco.MjModel.from_xml_string(enhanced_xml, mesh_assets)
    data = mujoco.MjData(model)
    return model, data


def save_video(frames, filename, fps=30):
    path = os.path.join(PLOTS_DIR, filename)
    try:
        import mediapy as media
        media.write_video(path, frames, fps=fps)
    except ImportError:
        import imageio
        imageio.mimwrite(path, frames, fps=fps)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {filename} ({len(frames)} frames, {len(frames)/fps:.1f}s, {size_kb:.0f}KB)")


def make_camera(azimuth=145, elevation=-25, distance=1.2,
                lookat=(0.1, 0.0, 0.25)):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.distance = distance
    cam.lookat[:] = lookat
    return cam


def lerp_angles(q_start, q_end, n_steps):
    t = np.linspace(0, 1, n_steps).reshape(-1, 1)
    return q_start + t * (q_end - q_start)


def smooth_step(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


# ==================== Demo 1: Rotating Overview ====================

def demo_overview(model, data, renderer):
    print("\n[Demo 1] Rotating camera overview...")
    poses = [
        np.zeros(6),
        np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0]),
        np.array([0.8, -0.3, 0.5, -0.2, 0.6, -0.4]),
        np.array([-0.5, 0.2, -0.3, 0.5, -0.4, 0.8]),
    ]
    frames = []
    n_rot = 90

    for p_idx, q in enumerate(poses):
        if p_idx > 0:
            traj = lerp_angles(poses[p_idx - 1], q, 30)
            for qi in traj:
                set_joint_angles(model, data, qi)
                cam = make_camera(azimuth=135, elevation=-25, distance=1.2)
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render().copy())

        set_joint_angles(model, data, q)
        for i in range(n_rot):
            az = 135 + (360 * i / n_rot)
            cam = make_camera(azimuth=az, elevation=-25, distance=1.2)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    save_video(frames, "demo1_overview.mp4", fps=30)
    return frames


# ==================== Demo 2: Joint Sweep ====================

def demo_joint_sweep(model, data, renderer):
    print("\n[Demo 2] Joint-by-joint sweep...")
    frames = []
    n_sweep = 50

    for j in range(NUM_JOINTS):
        q_base = np.zeros(6)
        # 0 -> +80 -> -80 -> 0
        angles = []
        for i in range(n_sweep):
            angles.append(smooth_step(i / n_sweep) * 1.4)
        for i in range(n_sweep * 2):
            angles.append(1.4 - smooth_step(i / (n_sweep * 2)) * 2.8)
        for i in range(n_sweep):
            angles.append(-1.4 + smooth_step(i / n_sweep) * 1.4)

        for a in angles:
            q = q_base.copy()
            q[j] = a
            set_joint_angles(model, data, q)
            cam = make_camera(azimuth=145, elevation=-22, distance=1.15)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    save_video(frames, "demo2_joint_sweep.mp4", fps=30)
    return frames


# ==================== Demo 3: Circle Trajectory ====================

def demo_circle(model, data, renderer):
    print("\n[Demo 3] Circle trajectory tracking...")

    q_ref = np.array([0.0, -0.3, 0.5, 0.0, 0.3, 0.0])
    T_ref = forward_kinematics(q_ref, use_urdf=True)
    ref_rot = T_ref[:3, :3]
    ref_pos = T_ref[:3, 3]

    center = ref_pos.copy()
    radius = 0.06
    n_pts = 120

    frames = []
    q_prev = q_ref.copy()

    for i in range(n_pts):
        t = 2 * np.pi * i / n_pts
        target_pos = center.copy()
        target_pos[0] += radius * np.cos(t)
        target_pos[2] += radius * np.sin(t)

        T_target = np.eye(4)
        T_target[:3, :3] = ref_rot
        T_target[:3, 3] = target_pos

        r = inverse_kinematics(T_target, initial_guess=q_prev,
                               method="damped_least_squares",
                               use_urdf=True, max_iterations=200)
        set_joint_angles(model, data, r.joint_angles)

        # Slowly rotate camera for more dynamic view
        az = 155 + 30 * np.sin(2 * np.pi * i / n_pts)
        cam = make_camera(azimuth=az, elevation=-20, distance=1.0,
                          lookat=(center[0], center[1], center[2]))
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
        q_prev = r.joint_angles

    save_video(frames, "demo3_circle.mp4", fps=30)

    # Key frames
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for idx, ki in enumerate([0, 15, 30, 45, 60, 75, 90, 105]):
        ax = axes[idx // 4][idx % 4]
        ax.imshow(frames[ki])
        angle = ki * 360 // n_pts
        ax.set_title(f"{angle}\u00b0", fontsize=12)
        ax.axis("off")
    fig.suptitle("Circle Trajectory - Key Frames", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "demo3_circle_keyframes.png"), dpi=150)
    plt.close()
    print(f"  Saved: demo3_circle_keyframes.png")
    return frames


# ==================== Demo 4: Pick-and-Place ====================

def demo_pick_and_place(model, data, renderer):
    print("\n[Demo 4] Pick-and-place trajectory...")

    q_home = np.array([0.0, -0.3, 0.5, 0.0, 0.3, 0.0])
    T_home = forward_kinematics(q_home, use_urdf=True)
    home_rot = T_home[:3, :3]
    home_pos = T_home[:3, 3]

    waypoints = [
        ("Home",        home_pos,                                    0.0),
        ("Above Pick",  home_pos + np.array([-0.06, -0.03, 0.04]),  0.0),
        ("Pick Down",   home_pos + np.array([-0.06, -0.03, -0.02]), 0.0),
        ("Grasp",       home_pos + np.array([-0.06, -0.03, -0.02]), 0.03),
        ("Pick Up",     home_pos + np.array([-0.06, -0.03, 0.06]),  0.03),
        ("Transit",     home_pos + np.array([0.0, 0.0, 0.06]),      0.03),
        ("Above Place", home_pos + np.array([0.06, 0.03, 0.06]),    0.03),
        ("Place Down",  home_pos + np.array([0.06, 0.03, -0.02]),   0.03),
        ("Release",     home_pos + np.array([0.06, 0.03, -0.02]),   0.0),
        ("Place Up",    home_pos + np.array([0.06, 0.03, 0.06]),    0.0),
        ("Home",        home_pos,                                    0.0),
    ]

    # Solve IK for all waypoints
    q_waypoints = []
    grip_values = []
    q_prev = q_home.copy()
    for name, pos, grip in waypoints:
        T = np.eye(4)
        T[:3, :3] = home_rot
        T[:3, 3] = pos
        r = inverse_kinematics(T, initial_guess=q_prev,
                               method="damped_least_squares",
                               use_urdf=True, max_iterations=300)
        q_waypoints.append(r.joint_angles)
        grip_values.append(grip)
        q_prev = r.joint_angles

    frames = []
    fps_per_seg = 40
    pause = 8

    for seg in range(len(q_waypoints) - 1):
        traj = lerp_angles(q_waypoints[seg], q_waypoints[seg + 1], fps_per_seg)
        # Interpolate gripper
        g_start, g_end = grip_values[seg], grip_values[seg + 1]

        for fi, qi in enumerate(traj):
            t = fi / fps_per_seg
            g = g_start + t * (g_end - g_start)
            data.qpos[:6] = qi
            data.qpos[6] = g
            data.qpos[7] = g
            mujoco.mj_forward(model, data)

            cam = make_camera(azimuth=150, elevation=-18, distance=1.05,
                              lookat=(home_pos[0], home_pos[1], home_pos[2]))
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

        # Pause at grasp/release
        name_next = waypoints[seg + 1][0]
        if name_next in ("Grasp", "Release", "Pick Down", "Place Down"):
            for _ in range(pause):
                frames.append(frames[-1].copy())

    save_video(frames, "demo4_pick_and_place.mp4", fps=30)

    # Key frames
    n_wp = len(waypoints)
    total = len(frames)
    fig, axes = plt.subplots(1, n_wp, figsize=(2.2 * n_wp, 3))
    for i in range(n_wp):
        fi = min(int(i * total / n_wp), total - 1)
        axes[i].imshow(frames[fi])
        axes[i].set_title(waypoints[i][0], fontsize=7)
        axes[i].axis("off")
    fig.suptitle("Pick-and-Place Trajectory", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "demo4_pick_place_keyframes.png"), dpi=150)
    plt.close()
    print(f"  Saved: demo4_pick_place_keyframes.png")
    return frames


# ==================== Demo 5: Workspace Exploration ====================

def demo_workspace(model, data, renderer):
    print("\n[Demo 5] Workspace exploration...")
    rng = np.random.RandomState(123)
    n_poses = 16
    fps_trans = 35

    configs = [np.zeros(6)]
    for _ in range(n_poses):
        configs.append(rng.uniform(-1.5, 1.5, 6))
    configs.append(np.zeros(6))

    frames = []
    for i in range(len(configs) - 1):
        traj = lerp_angles(configs[i], configs[i + 1], fps_trans)
        for fi, qi in enumerate(traj):
            set_joint_angles(model, data, qi)
            az = 140 + i * 12 + fi * 0.3
            cam = make_camera(azimuth=az, elevation=-22, distance=1.15)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())

    save_video(frames, "demo5_workspace.mp4", fps=30)
    return frames


# ==================== Demo 6: Multi-View ====================

def demo_multiview(model, data, renderer):
    print("\n[Demo 6] Multi-view circle trajectory...")

    q_ref = np.array([0.0, -0.3, 0.5, 0.0, 0.3, 0.0])
    T_ref = forward_kinematics(q_ref, use_urdf=True)
    ref_rot = T_ref[:3, :3]
    ref_pos = T_ref[:3, 3]
    center = ref_pos.copy()
    radius = 0.05
    n_pts = 90

    cam_params = [
        ("Front", 180, -15),
        ("Side",  90,  -15),
        ("Top",   180, -89),
        ("3/4",   145, -25),
    ]

    all_views = {n: [] for n, _, _ in cam_params}
    q_prev = q_ref.copy()

    half_w, half_h = VIDEO_W // 2, VIDEO_H // 2
    # Use a smaller renderer for each view
    mini_renderer = mujoco.Renderer(model, height=half_h, width=half_w)

    for i in range(n_pts):
        t = 2 * np.pi * i / n_pts
        tp = center.copy()
        tp[0] += radius * np.cos(t)
        tp[2] += radius * np.sin(t)

        T_target = np.eye(4)
        T_target[:3, :3] = ref_rot
        T_target[:3, 3] = tp

        r = inverse_kinematics(T_target, initial_guess=q_prev,
                               method="damped_least_squares",
                               use_urdf=True, max_iterations=200)
        set_joint_angles(model, data, r.joint_angles)
        q_prev = r.joint_angles

        for name, az, el in cam_params:
            cam = make_camera(azimuth=az, elevation=el, distance=1.0,
                              lookat=tuple(center))
            mini_renderer.update_scene(data, camera=cam)
            all_views[name].append(mini_renderer.render().copy())

    mini_renderer.close()

    # Combine into 2x2 grid
    names = [n for n, _, _ in cam_params]
    grid_frames = []
    for i in range(n_pts):
        top = np.concatenate([all_views[names[0]][i], all_views[names[1]][i]], axis=1)
        bot = np.concatenate([all_views[names[2]][i], all_views[names[3]][i]], axis=1)
        grid_frames.append(np.concatenate([top, bot], axis=0))

    save_video(grid_frames, "demo6_multiview.mp4", fps=30)

    # Snapshot
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for idx, (name, _, _) in enumerate(cam_params):
        ax = axes[idx // 2][idx % 2]
        ax.imshow(all_views[name][0])
        ax.set_title(f"{name} View", fontsize=12)
        ax.axis("off")
    fig.suptitle("Multi-View Robot Arm", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "demo6_multiview_snapshot.png"), dpi=150)
    plt.close()
    print(f"  Saved: demo6_multiview_snapshot.png")
    return grid_frames


# ==================== Main ====================

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.set_printoptions(precision=4, suppress=True)

    print("Loading enhanced MuJoCo model (with lighting, floor, colors)...")
    model, data = load_enhanced_model()
    print(f"  Model loaded: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    print(f"Creating renderer ({VIDEO_W}x{VIDEO_H})...")
    renderer = mujoco.Renderer(model, height=VIDEO_H, width=VIDEO_W)

    print("\n" + "=" * 60)
    print("  Generating Robot Arm Visualization Demos")
    print("=" * 60)

    demo_overview(model, data, renderer)
    demo_joint_sweep(model, data, renderer)
    demo_circle(model, data, renderer)
    demo_pick_and_place(model, data, renderer)
    demo_workspace(model, data, renderer)

    # Multi-view uses its own smaller renderer
    demo_multiview(model, data, renderer)

    renderer.close()

    print("\n" + "=" * 60)
    print("  All demos complete!")
    print(f"  Videos saved to: {PLOTS_DIR}")
    print("=" * 60)

    files = sorted(os.listdir(PLOTS_DIR))
    videos = [f for f in files if f.startswith("demo") and f.endswith(".mp4")]
    print(f"\n  Generated videos ({len(videos)}):")
    for v in videos:
        size_kb = os.path.getsize(os.path.join(PLOTS_DIR, v)) / 1024
        print(f"    {v:35s} {size_kb:>7.0f} KB")


if __name__ == "__main__":
    main()
