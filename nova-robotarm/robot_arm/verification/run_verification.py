"""
Comprehensive verification of the kinematics module against MuJoCo ground truth.

Tests:
    A. Forward Kinematics (URDF vs MuJoCo, DH vs MuJoCo)
    B. Jacobian (analytical vs MuJoCo mj_jac)
    C. IK round-trip (MuJoCo FK → our IK → MuJoCo FK)
    D. Trajectory tracking (circle + line in task space)

Generates matplotlib plots in robot_arm/verification/plots/.

Usage:
    python -m robot_arm.verification.run_verification [--render] [--n-random 50]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Add project root to path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot_arm.kinematics.kinematics import (
    forward_kinematics,
    compute_jacobian,
    inverse_kinematics,
    get_end_effector_pose,
    rotation_matrix_to_euler,
    _rot_to_axis_angle,
    NUM_JOINTS,
)
from robot_arm.verification.mujoco_loader import (
    load_mujoco_model,
    set_joint_angles,
    get_ee_pose,
    get_mujoco_jacobian,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def ensure_plots_dir():
    os.makedirs(PLOTS_DIR, exist_ok=True)


def sep(title):
    w = 72
    pad = max((w - len(title) - 2) // 2, 0)
    print("\n" + "=" * pad + f" {title} " + "=" * pad)


# --------------- Test Configuration Generation ---------------

def generate_configs(n_random=50, seed=42):
    """Generate a comprehensive set of test joint configurations."""
    configs = []
    labels = []

    # 1. Zero config
    configs.append(np.zeros(6))
    labels.append("q=0")

    # 2. Single-joint sweeps
    for j in range(6):
        for val in [-2.0, -1.0, 0.5, 1.0, 2.0]:
            q = np.zeros(6)
            q[j] = val
            configs.append(q)
            labels.append(f"j{j+1}={val:.1f}")

    # 3. Specific config from kinematics tests
    configs.append(np.array([0.3, -0.5, 0.8, -0.2, 0.6, -0.4]))
    labels.append("test_config")

    # 4. Random configs
    rng = np.random.RandomState(seed)
    for i in range(n_random):
        q = rng.uniform(-np.pi, np.pi, 6)
        configs.append(q)
        labels.append(f"rand_{i}")

    return configs, labels


# --------------- A. FK Verification ---------------

def verify_fk(configs, labels, model, data):
    sep("A. Forward Kinematics Verification")

    pos_err_urdf = []
    ori_err_urdf = []
    pos_err_dh = []
    ori_err_dh = []
    mj_positions = []
    urdf_positions = []
    dh_positions = []

    for i, q in enumerate(configs):
        # MuJoCo ground truth
        set_joint_angles(model, data, q)
        mj_pos, mj_rot = get_ee_pose(model, data)

        # Our URDF FK
        T_u = forward_kinematics(q, use_urdf=True)
        pe_u = np.linalg.norm(T_u[:3, 3] - mj_pos)
        oe_u = np.linalg.norm(_rot_to_axis_angle(T_u[:3, :3] @ mj_rot.T))

        # Our DH FK
        T_d = forward_kinematics(q, use_urdf=False)
        pe_d = np.linalg.norm(T_d[:3, 3] - mj_pos)
        oe_d = np.linalg.norm(_rot_to_axis_angle(T_d[:3, :3] @ mj_rot.T))

        pos_err_urdf.append(pe_u)
        ori_err_urdf.append(oe_u)
        pos_err_dh.append(pe_d)
        ori_err_dh.append(oe_d)
        mj_positions.append(mj_pos)
        urdf_positions.append(T_u[:3, 3].copy())
        dh_positions.append(T_d[:3, 3].copy())

    pos_err_urdf = np.array(pos_err_urdf)
    ori_err_urdf = np.array(ori_err_urdf)
    pos_err_dh = np.array(pos_err_dh)
    ori_err_dh = np.array(ori_err_dh)

    # Print summary
    print(f"\nURDF mode vs MuJoCo:")
    print(f"  Position error: max={pos_err_urdf.max():.2e}  mean={pos_err_urdf.mean():.2e}  (m)")
    print(f"  Orient. error:  max={ori_err_urdf.max():.2e}  mean={ori_err_urdf.mean():.2e}  (rad)")
    urdf_pass = pos_err_urdf.max() < 1e-4
    print(f"  URDF FK match: {'PASS' if urdf_pass else 'FAIL'}")

    print(f"\nDH mode vs MuJoCo (structural discrepancy expected):")
    print(f"  Position error: max={pos_err_dh.max():.4f}  mean={pos_err_dh.mean():.4f}  (m)")
    print(f"  Orient. error:  max={ori_err_dh.max():.4f}  mean={ori_err_dh.mean():.4f}  (rad)")

    # --- Plot 1: FK Position Error ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    ax1.semilogy(pos_err_urdf, "b.", markersize=4, label="URDF mode")
    ax1.axhline(1e-6, color="g", ls="--", alpha=0.5, label="1e-6 m")
    ax1.set_xlabel("Configuration index")
    ax1.set_ylabel("Position error (m)")
    ax1.set_title("URDF FK vs MuJoCo - Position Error")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(pos_err_dh * 100, bins=20, color="orange", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Position error (cm)")
    ax2.set_ylabel("Count")
    ax2.set_title("DH FK vs MuJoCo - Position Error Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fk_position_error.png"), dpi=150)
    plt.close()

    # --- Plot 2: FK Orientation Error ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].semilogy(ori_err_urdf, "b.", markersize=4, label="URDF mode")
    axes[0].axhline(1e-6, color="g", ls="--", alpha=0.5, label="1e-6 rad")
    axes[0].set_xlabel("Configuration index")
    axes[0].set_ylabel("Orientation error (rad)")
    axes[0].set_title("URDF FK vs MuJoCo - Orientation Error")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.degrees(ori_err_dh), bins=20, color="orange", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Orientation error (deg)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("DH FK vs MuJoCo - Orientation Error Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fk_orientation_error.png"), dpi=150)
    plt.close()

    # --- Plot 3: 3D Comparison ---
    mj_pos_arr = np.array(mj_positions)
    urdf_pos_arr = np.array(urdf_positions)
    dh_pos_arr = np.array(dh_positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(mj_pos_arr[:, 0], mj_pos_arr[:, 1], mj_pos_arr[:, 2],
               c="red", s=10, alpha=0.5, label="MuJoCo")
    ax.scatter(urdf_pos_arr[:, 0], urdf_pos_arr[:, 1], urdf_pos_arr[:, 2],
               c="blue", s=10, alpha=0.5, label="URDF FK")
    ax.scatter(dh_pos_arr[:, 0], dh_pos_arr[:, 1], dh_pos_arr[:, 2],
               c="orange", s=10, alpha=0.5, label="DH FK")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Positions: MuJoCo vs URDF vs DH")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "fk_3d_comparison.png"), dpi=150)
    plt.close()

    print(f"\n  Plots saved: fk_position_error.png, fk_orientation_error.png, fk_3d_comparison.png")
    return urdf_pass, pos_err_urdf, pos_err_dh


# --------------- B. Jacobian Verification ---------------

def verify_jacobian(configs, labels, model, data):
    sep("B. Jacobian Verification")

    max_errs_urdf = []
    max_errs_dh = []
    frob_errs_urdf = []

    # Use a representative config for the heatmap
    q_repr = np.array([0.3, -0.5, 0.8, -0.2, 0.6, -0.4])

    for q in configs:
        set_joint_angles(model, data, q)
        J_mj = get_mujoco_jacobian(model, data)
        J_urdf = compute_jacobian(q, use_urdf=True)
        J_dh = compute_jacobian(q, use_urdf=False)

        max_errs_urdf.append(np.max(np.abs(J_urdf - J_mj)))
        max_errs_dh.append(np.max(np.abs(J_dh - J_mj)))
        frob_errs_urdf.append(np.linalg.norm(J_urdf - J_mj))

    max_errs_urdf = np.array(max_errs_urdf)
    max_errs_dh = np.array(max_errs_dh)
    frob_errs_urdf = np.array(frob_errs_urdf)

    print(f"\nURDF Jacobian vs MuJoCo:")
    print(f"  Max element error: max={max_errs_urdf.max():.2e}  mean={max_errs_urdf.mean():.2e}")
    jac_pass = max_errs_urdf.max() < 1e-4
    print(f"  Jacobian match: {'PASS' if jac_pass else 'FAIL'}")

    print(f"\nDH Jacobian vs MuJoCo:")
    print(f"  Max element error: max={max_errs_dh.max():.4f}  mean={max_errs_dh.mean():.4f}")

    # --- Plot 4: Jacobian Error Heatmap ---
    set_joint_angles(model, data, q_repr)
    J_mj = get_mujoco_jacobian(model, data)
    J_urdf = compute_jacobian(q_repr, use_urdf=True)
    J_dh = compute_jacobian(q_repr, use_urdf=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im1 = axes[0].imshow(np.abs(J_urdf - J_mj), aspect="auto", cmap="hot")
    axes[0].set_title("URDF Jacobian Error (|J_ours - J_mujoco|)")
    axes[0].set_xlabel("Joint")
    axes[0].set_ylabel("Row (v_xyz / w_xyz)")
    axes[0].set_xticks(range(6))
    axes[0].set_xticklabels([f"J{i+1}" for i in range(6)])
    axes[0].set_yticks(range(6))
    axes[0].set_yticklabels(["vx", "vy", "vz", "wx", "wy", "wz"])
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(np.abs(J_dh - J_mj), aspect="auto", cmap="hot")
    axes[1].set_title("DH Jacobian Error (|J_ours - J_mujoco|)")
    axes[1].set_xlabel("Joint")
    axes[1].set_ylabel("Row (v_xyz / w_xyz)")
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels([f"J{i+1}" for i in range(6)])
    axes[1].set_yticks(range(6))
    axes[1].set_yticklabels(["vx", "vy", "vz", "wx", "wy", "wz"])
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "jacobian_heatmap.png"), dpi=150)
    plt.close()

    # --- Plot 5: Jacobian Error Statistics ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot([max_errs_urdf, max_errs_dh],
                    tick_labels=["URDF mode", "DH mode"],
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightyellow")
    ax.set_ylabel("Max element-wise error")
    ax.set_title("Jacobian Error Statistics Across Configurations")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "jacobian_error_stats.png"), dpi=150)
    plt.close()

    print(f"  Plots saved: jacobian_heatmap.png, jacobian_error_stats.png")
    return jac_pass


# --------------- C. IK Round-Trip Verification ---------------

def verify_ik(configs, labels, model, data):
    sep("C. IK Round-Trip Verification")

    # Use a subset (first 20 configs for speed)
    test_configs = configs[:min(20, len(configs))]
    rng = np.random.RandomState(42)

    results = []  # (label, method, mode, success, iters, pos_err, ori_err)

    for idx, q_gt in enumerate(test_configs):
        # Get target pose from MuJoCo
        set_joint_angles(model, data, q_gt)
        mj_pos, mj_rot = get_ee_pose(model, data)

        # Build target as 4x4 transform
        T_target = np.eye(4)
        T_target[:3, :3] = mj_rot
        T_target[:3, 3] = mj_pos

        for method in ["damped_least_squares", "newton_raphson"]:
            short_method = "DLS" if "damped" in method else "NR"

            # URDF mode - perturbed guess
            ig = q_gt + rng.uniform(-0.3, 0.3, 6)
            r = inverse_kinematics(T_target, initial_guess=ig,
                                   method=method, use_urdf=True,
                                   max_iterations=500)
            # Verify in MuJoCo
            set_joint_angles(model, data, r.joint_angles)
            rec_pos, rec_rot = get_ee_pose(model, data)
            pe = np.linalg.norm(mj_pos - rec_pos)
            oe = np.linalg.norm(_rot_to_axis_angle(mj_rot @ rec_rot.T))

            results.append((labels[idx], short_method, "URDF", r.success,
                            r.iterations, pe, oe))

    # Print summary table
    successes = sum(1 for r in results if r[3])
    total = len(results)
    pe_arr = np.array([r[5] for r in results if r[3]])
    oe_arr = np.array([r[6] for r in results if r[3]])

    print(f"\nIK Round-trip Results (URDF mode, verified in MuJoCo):")
    print(f"  Success rate: {successes}/{total} ({100*successes/total:.1f}%)")
    if len(pe_arr) > 0:
        print(f"  Position error: max={pe_arr.max():.2e}  mean={pe_arr.mean():.2e} (m)")
        print(f"  Orient. error:  max={oe_arr.max():.2e}  mean={oe_arr.mean():.2e} (rad)")

    # Pass if >=80% converge and all converged solutions have < 1mm error
    ik_pass = (successes / total >= 0.8) and (len(pe_arr) == 0 or pe_arr.max() < 1e-3)

    # Separate DLS and NR results
    dls_results = [r for r in results if r[1] == "DLS"]
    nr_results = [r for r in results if r[1] == "NR"]

    # --- Plot 7: IK Round-Trip Position Error ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dls_pe = [r[5] for r in dls_results]
    nr_pe = [r[5] for r in nr_results]
    dls_iters = [r[4] for r in dls_results]
    nr_iters = [r[4] for r in nr_results]

    axes[0].semilogy(dls_pe, "bo", markersize=4, label="DLS", alpha=0.7)
    axes[0].semilogy(nr_pe, "rs", markersize=4, label="NR", alpha=0.7)
    axes[0].axhline(1e-4, color="g", ls="--", alpha=0.5, label="0.1mm")
    axes[0].set_xlabel("Configuration index")
    axes[0].set_ylabel("Position error after round-trip (m)")
    axes[0].set_title("IK Round-Trip Position Error (MuJoCo verified)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(np.arange(len(dls_iters)) - 0.2, dls_iters, 0.4,
                label="DLS", color="steelblue", alpha=0.7)
    axes[1].bar(np.arange(len(nr_iters)) + 0.2, nr_iters, 0.4,
                label="NR", color="indianred", alpha=0.7)
    axes[1].set_xlabel("Configuration index")
    axes[1].set_ylabel("Iterations")
    axes[1].set_title("IK Convergence: Iteration Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ik_roundtrip.png"), dpi=150)
    plt.close()

    print(f"  IK verification: {'PASS' if ik_pass else 'FAIL'}")
    print(f"  Plot saved: ik_roundtrip.png")
    return ik_pass


# --------------- D. Trajectory Tracking ---------------

def plan_circle_trajectory(center, radius, n_points, orientation_rot):
    """Generate a circle trajectory in the XZ plane."""
    targets = []
    for i in range(n_points):
        t = 2 * np.pi * i / n_points
        pos = center.copy()
        pos[0] += radius * np.cos(t)
        pos[2] += radius * np.sin(t)
        T = np.eye(4)
        T[:3, :3] = orientation_rot
        T[:3, 3] = pos
        targets.append(T)
    return targets


def plan_line_trajectory(start_pos, end_pos, n_points, orientation_rot):
    """Generate a straight line trajectory."""
    targets = []
    for i in range(n_points):
        t = i / max(n_points - 1, 1)
        pos = start_pos + t * (end_pos - start_pos)
        T = np.eye(4)
        T[:3, :3] = orientation_rot
        T[:3, 3] = pos
        targets.append(T)
    return targets


def verify_trajectory(model, data):
    sep("D. Trajectory Tracking Verification")

    # Get a reference orientation from a known reachable config
    q_ref = np.array([0.0, -0.3, 0.5, 0.0, 0.3, 0.0])
    T_ref = forward_kinematics(q_ref, use_urdf=True)
    ref_rot = T_ref[:3, :3]
    ref_pos = T_ref[:3, 3]

    # Circle trajectory centered near the reference position
    center = ref_pos.copy()
    radius = 0.04
    n_pts = 60
    circle_targets = plan_circle_trajectory(center, radius, n_pts, ref_rot)

    # Line trajectory
    start = ref_pos + np.array([-0.05, 0.0, 0.02])
    end = ref_pos + np.array([0.05, 0.0, -0.02])
    line_targets = plan_line_trajectory(start, end, 40, ref_rot)

    all_results = {}

    for traj_name, targets in [("circle", circle_targets), ("line", line_targets)]:
        desired_pos = []
        actual_pos = []
        pos_errors = []
        ori_errors = []
        joint_angles_list = []
        ik_successes = []

        q_prev = q_ref.copy()  # warm start

        for T_target in targets:
            # Solve IK with warm start
            r = inverse_kinematics(T_target, initial_guess=q_prev,
                                   method="damped_least_squares",
                                   use_urdf=True, max_iterations=500)
            q_sol = r.joint_angles
            ik_successes.append(r.success)

            # Verify in MuJoCo
            set_joint_angles(model, data, q_sol)
            mj_pos, mj_rot = get_ee_pose(model, data)

            desired_pos.append(T_target[:3, 3].copy())
            actual_pos.append(mj_pos.copy())
            pe = np.linalg.norm(T_target[:3, 3] - mj_pos)
            oe = np.linalg.norm(_rot_to_axis_angle(T_target[:3, :3] @ mj_rot.T))
            pos_errors.append(pe)
            ori_errors.append(oe)
            joint_angles_list.append(q_sol.copy())

            q_prev = q_sol  # warm start for next

        desired_pos = np.array(desired_pos)
        actual_pos = np.array(actual_pos)
        pos_errors = np.array(pos_errors)
        ori_errors = np.array(ori_errors)
        joint_angles_arr = np.array(joint_angles_list)

        n_success = sum(ik_successes)
        print(f"\n  {traj_name.upper()} trajectory ({len(targets)} waypoints):")
        print(f"    IK success: {n_success}/{len(targets)}")
        print(f"    Position error: max={pos_errors.max():.2e}  "
              f"mean={pos_errors.mean():.2e}  rms={np.sqrt(np.mean(pos_errors**2)):.2e} (m)")
        print(f"    Orient. error:  max={ori_errors.max():.2e}  "
              f"mean={ori_errors.mean():.2e} (rad)")

        all_results[traj_name] = {
            "desired": desired_pos, "actual": actual_pos,
            "pos_errors": pos_errors, "ori_errors": ori_errors,
            "joints": joint_angles_arr, "ik_success": ik_successes,
        }

    # --- Plot 9: 3D Trajectory ---
    fig = plt.figure(figsize=(14, 6))
    for idx, (name, res) in enumerate(all_results.items()):
        ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
        d = res["desired"]
        a = res["actual"]
        ax.plot(d[:, 0], d[:, 1], d[:, 2], "b--", linewidth=1.5, label="Desired")
        ax.plot(a[:, 0], a[:, 1], a[:, 2], "r-", linewidth=1.5, label="Actual (MuJoCo)")
        ax.scatter(d[0, 0], d[0, 1], d[0, 2], c="green", s=50, marker="^", label="Start")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(f"{name.capitalize()} Trajectory Tracking")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trajectory_3d.png"), dpi=150)
    plt.close()

    # --- Plot 10: Position Error Over Time ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        n = len(res["pos_errors"])
        ax.plot(range(n), res["pos_errors"] * 1000, "b-", linewidth=1.5, label="Position error")
        ax.axhline(0.1, color="g", ls="--", alpha=0.5, label="0.1 mm")
        ax.set_xlabel("Waypoint index")
        ax.set_ylabel("Position error (mm)")
        ax.set_title(f"{name.capitalize()} - Position Tracking Error")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trajectory_error.png"), dpi=150)
    plt.close()

    # --- Plot 11: Joint Angles ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    name = "circle"
    res = all_results[name]
    for j in range(6):
        ax = axes[j // 3][j % 3]
        ax.plot(np.degrees(res["joints"][:, j]), linewidth=1.5)
        ax.set_xlabel("Waypoint")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(f"Joint {j+1}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Joint Angles During Circle Trajectory", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "trajectory_joints.png"), dpi=150)
    plt.close()

    # Check overall pass
    all_pe = np.concatenate([r["pos_errors"] for r in all_results.values()])
    traj_pass = all_pe.max() < 1e-3  # sub-mm
    print(f"\n  Trajectory tracking: {'PASS' if traj_pass else 'FAIL'}")
    print(f"  Plots saved: trajectory_3d.png, trajectory_error.png, trajectory_joints.png")
    return traj_pass, all_results


# --------------- E. Optional MuJoCo Rendering ---------------

def render_trajectory(model, data, all_results):
    sep("E. MuJoCo Rendering")
    try:
        import mujoco
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception as e:
        print(f"  Renderer not available: {e}")
        return

    # Render circle trajectory
    res = all_results.get("circle")
    if res is None:
        print("  No circle trajectory data to render.")
        renderer.close()
        return

    frames = []
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.2
    camera.azimuth = 135
    camera.elevation = -25
    camera.lookat[:] = [0.2, 0.0, 0.3]

    for q in res["joints"]:
        set_joint_angles(model, data, q)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        frames.append(frame.copy())

    renderer.close()

    try:
        import mediapy as media
        video_path = os.path.join(PLOTS_DIR, "trajectory.mp4")
        media.write_video(video_path, frames, fps=15)
        print(f"  Video saved: trajectory.mp4 ({len(frames)} frames)")
    except ImportError:
        # Fallback: save key frames as images
        for i in range(0, len(frames), max(len(frames) // 8, 1)):
            path = os.path.join(PLOTS_DIR, f"frame_{i:03d}.png")
            plt.imsave(path, frames[i])
        print(f"  Saved {min(8, len(frames))} key frames as PNGs (mediapy not available)")


# --------------- Main ---------------

def main():
    parser = argparse.ArgumentParser(description="Verify kinematics module against MuJoCo")
    parser.add_argument("--render", action="store_true", help="Render MuJoCo visualization")
    parser.add_argument("--n-random", type=int, default=50, help="Number of random configs")
    args = parser.parse_args()

    ensure_plots_dir()
    np.set_printoptions(precision=6, suppress=True, linewidth=120)

    print("Loading MuJoCo model...")
    model, data = load_mujoco_model()
    print(f"  Model loaded: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")

    configs, labels = generate_configs(n_random=args.n_random)
    print(f"  Generated {len(configs)} test configurations")

    # Run all verifications
    fk_pass, _, _ = verify_fk(configs, labels, model, data)
    jac_pass = verify_jacobian(configs, labels, model, data)
    ik_pass = verify_ik(configs, labels, model, data)
    traj_pass, traj_results = verify_trajectory(model, data)

    if args.render:
        render_trajectory(model, data, traj_results)

    # Final summary
    sep("FINAL SUMMARY")
    results = {
        "FK (URDF vs MuJoCo)": fk_pass,
        "Jacobian (URDF vs MuJoCo)": jac_pass,
        "IK Round-Trip": ik_pass,
        "Trajectory Tracking": traj_pass,
    }
    all_pass = True
    for name, ok in results.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  {name:35s} {tag}")
        if not ok:
            all_pass = False

    print()
    print(f"  Overall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print(f"\n  All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
