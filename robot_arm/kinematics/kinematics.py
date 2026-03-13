"""
Forward Kinematics (FK) and Inverse Kinematics (IK) module for a 6-DOF robot arm.

Supports two FK modes:
    - DH mode (default): Modified DH parameters from Inv_Dyn_2.py
    - URDF mode: Joint transforms from arm.urdf (SolidWorks CAD export)

The URDF mode preserves exact CAD geometry and is more accurate.
DH mode is provided for compatibility with the dynamics module.

Dependencies:
    numpy
"""

import numpy as np
from typing import NamedTuple


NUM_JOINTS = 6

# ==================== DH Parameters (from Inv_Dyn_2.py) ====================

DH_ALPHA = np.array([
    0.0,                    # Joint 1
    np.pi / 2,             # Joint 2
    0.0,                    # Joint 3
    0.0,                    # Joint 4
    -np.pi / 2,            # Joint 5
    -np.pi / 2,            # Joint 6
])

DH_A = np.array([
    0.0,                    # Joint 1
    0.0,                    # Joint 2
    0.3846,                 # Joint 3
    0.3179,                 # Joint 4
    0.0715,                 # Joint 5
    0.0,                    # Joint 6
])

DH_D = np.array([
    0.1367,                 # Joint 1
    0.0,                    # Joint 2
    0.0,                    # Joint 3
    0.0,                    # Joint 4
    0.0,                    # Joint 5
    0.026,                  # Joint 6
])

DH_THETA_OFFSET = np.array([
    0.0,                                    # Joint 1
    173.01 * np.pi / 180.0,                # Joint 2: +173.01 deg
    -157.51 * np.pi / 180.0,               # Joint 3: -157.51 deg
    -15.5 * np.pi / 180.0,                 # Joint 4: -15.5 deg
    -np.pi / 2,                             # Joint 5: -90 deg
    0.0,                                    # Joint 6
])

# ==================== URDF Joint Transforms (from arm.urdf) ====================

_URDF_JOINT_XYZ = [
    np.array([0.0, 0.0, 0.1281]),
    np.array([0.0, 0.0, 0.0]),
    np.array([0.37425, 0.0, 0.0]),
    np.array([0.30395, 0.00032772, 0.0]),
    np.array([0.080376, 0.0018417, -0.0020003]),
    np.array([0.0020919, 0.029927, 0.0]),
]

_URDF_JOINT_RPY = [
    np.array([0.0, 0.0, 0.0]),
    np.array([-1.5708, 0.0, 3.1397]),
    np.array([0.0, 0.0, -2.8166]),
    np.array([0.0, 0.0, -0.3307]),
    np.array([-1.5708, -1.5689, 0.0056773]),
    np.array([-1.5708, 0.0, -0.0019006]),
]

_URDF_GRIPPER_XYZ = np.array([-0.0016279, 0.0, 0.064947])
_URDF_GRIPPER_RPY = np.array([0.0, 0.0, -0.048473])

# ==================== Physical Parameters ====================

JOINT_LIMITS_LOWER = np.array([-np.pi, -np.pi, -np.pi, -np.pi, -np.pi, -np.pi])
JOINT_LIMITS_UPPER = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])


class IKResult(NamedTuple):
    """Container for inverse kinematics results."""
    joint_angles: np.ndarray
    success: bool
    position_error: float
    orientation_error: float
    iterations: int
    message: str


# ==================== Core Transform Helpers ====================

def _rpy_to_rotation(roll, pitch, yaw):
    """Rotation matrix from roll-pitch-yaw (URDF convention: R = Rz*Ry*Rx)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])


def _make_transform(R, p):
    """Build 4x4 homogeneous transform from 3x3 rotation and 3-vector."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _rz_transform(theta):
    """4x4 rotation about z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] = c
    return T


def dh_transform(alpha, a, d, theta):
    """Modified DH 4x4 homogeneous transform. Matches DHTrans() in Inv_Dyn_2.py."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,      -st,      0.0,   a      ],
        [st * ca,  ct * ca, -sa,  -sa * d  ],
        [st * sa,  ct * sa,  ca,   ca * d  ],
        [0.0,      0.0,      0.0,  1.0     ],
    ])


def _rot_to_axis_angle(R):
    """Convert rotation matrix to axis-angle vector (angle * axis)."""
    trace_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(trace_val)
    if abs(angle) < 1e-10:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-6:
        M = R + np.eye(3)
        norms = np.linalg.norm(M, axis=0)
        best = np.argmax(norms)
        return (M[:, best] / norms[best]) * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2.0 * np.sin(angle)) * angle


# ==================== Precompute URDF fixed transforms ====================

def _build_urdf_fixed_transforms():
    """Precompute fixed (non-joint-dependent) transforms from URDF data."""
    fixed = []
    for i in range(NUM_JOINTS):
        r, p, y = _URDF_JOINT_RPY[i]
        R = _rpy_to_rotation(r, p, y)
        fixed.append(_make_transform(R, _URDF_JOINT_XYZ[i]))
    r, p, y = _URDF_GRIPPER_RPY
    R_g = _rpy_to_rotation(r, p, y)
    T_g = _make_transform(R_g, _URDF_GRIPPER_XYZ)
    return fixed, T_g


_URDF_FIXED, _URDF_GRIPPER_T = _build_urdf_fixed_transforms()


# ==================== Forward Kinematics ====================

def forward_kinematics(joint_angles, use_urdf=False):
    """
    Compute end-effector 4x4 homogeneous transform from joint angles.

    Parameters
    ----------
    joint_angles : (6,) array in radians
    use_urdf : bool, if True use URDF transforms instead of DH
    """
    q = np.asarray(joint_angles, dtype=float)
    assert q.shape == (NUM_JOINTS,), f"Expected {NUM_JOINTS} joints, got {q.shape}"

    T = np.eye(4)
    if use_urdf:
        for i in range(NUM_JOINTS):
            T = T @ _URDF_FIXED[i] @ _rz_transform(q[i])
        T = T @ _URDF_GRIPPER_T
    else:
        for i in range(NUM_JOINTS):
            T = T @ dh_transform(DH_ALPHA[i], DH_A[i], DH_D[i],
                                 q[i] + DH_THETA_OFFSET[i])
    return T


def get_all_transforms(joint_angles, use_urdf=False):
    """
    Return list of cumulative transforms.

    DH mode:   [T_base, T_01, ..., T_06]  (7 elements)
    URDF mode: [T_base, T_after_j1, ..., T_after_j6, T_ee]  (8 elements)
    """
    q = np.asarray(joint_angles, dtype=float)
    assert q.shape == (NUM_JOINTS,)

    transforms = [np.eye(4)]
    T = np.eye(4)
    if use_urdf:
        for i in range(NUM_JOINTS):
            T = T @ _URDF_FIXED[i] @ _rz_transform(q[i])
            transforms.append(T.copy())
        T = T @ _URDF_GRIPPER_T
        transforms.append(T.copy())
    else:
        for i in range(NUM_JOINTS):
            T = T @ dh_transform(DH_ALPHA[i], DH_A[i], DH_D[i],
                                 q[i] + DH_THETA_OFFSET[i])
            transforms.append(T.copy())
    return transforms


def _get_joint_axis_frames_urdf(joint_angles):
    """
    Get frames where z-axis is the joint rotation axis (URDF mode).

    For the geometric Jacobian, the joint axis is z of the frame AFTER
    the fixed transform but BEFORE the joint rotation.
    """
    q = np.asarray(joint_angles, dtype=float)
    frames = []
    T = np.eye(4)
    for i in range(NUM_JOINTS):
        T_pre = T @ _URDF_FIXED[i]
        frames.append(T_pre.copy())
        T = T_pre @ _rz_transform(q[i])
    return frames


def get_end_effector_position(joint_angles, use_urdf=False):
    """Return (x, y, z) end-effector position [m]."""
    return forward_kinematics(joint_angles, use_urdf)[:3, 3].copy()


def get_end_effector_pose(joint_angles, use_urdf=False):
    """Return (x, y, z, roll, pitch, yaw) of the end-effector."""
    T = forward_kinematics(joint_angles, use_urdf)
    return np.concatenate([T[:3, 3], rotation_matrix_to_euler(T[:3, :3])])


# ==================== Jacobian ====================

def compute_jacobian(joint_angles, use_urdf=False):
    """
    Compute 6x6 geometric Jacobian.  [v; w] = J * dq.

    For revolute joints:
        J_v_i = z_i x (o_ee - o_i)
        J_w_i = z_i
    """
    J = np.zeros((6, NUM_JOINTS))

    if use_urdf:
        axis_frames = _get_joint_axis_frames_urdf(joint_angles)
        o_n = forward_kinematics(joint_angles, use_urdf=True)[:3, 3]
        for i in range(NUM_JOINTS):
            z_i = axis_frames[i][:3, 2]
            o_i = axis_frames[i][:3, 3]
            J[:3, i] = np.cross(z_i, o_n - o_i)
            J[3:, i] = z_i
    else:
        transforms = get_all_transforms(joint_angles, use_urdf=False)
        o_n = transforms[-1][:3, 3]
        for i in range(NUM_JOINTS):
            # In MDH convention, z-axis of transforms[i+1] is the joint axis
            # because Rz*Tz preserves the z direction.
            z_i = transforms[i + 1][:3, 2]
            o_i = transforms[i + 1][:3, 3]
            J[:3, i] = np.cross(z_i, o_n - o_i)
            J[3:, i] = z_i
    return J


def compute_jacobian_numerical(joint_angles, use_urdf=False, delta=1e-7):
    """
    Numerical Jacobian via central finite differences (for verification).

    Uses rotation matrix log-map for angular part to match the geometric Jacobian.
    """
    q = np.asarray(joint_angles, dtype=float)
    J = np.zeros((6, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        qp = q.copy(); qp[i] += delta
        qm = q.copy(); qm[i] -= delta
        Tp = forward_kinematics(qp, use_urdf)
        Tm = forward_kinematics(qm, use_urdf)
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2.0 * delta)
        R_diff = Tp[:3, :3] @ Tm[:3, :3].T
        J[3:, i] = _rot_to_axis_angle(R_diff) / (2.0 * delta)
    return J


# ==================== Inverse Kinematics ====================

def _pose_error(q, target_pos, target_rot, use_urdf=False):
    """Compute 6D pose error [dp(3), dr(3)] and scalar norms."""
    T = forward_kinematics(q, use_urdf)
    dp = target_pos - T[:3, 3]
    R_err = target_rot @ T[:3, :3].T
    dr = _rot_to_axis_angle(R_err)
    return np.concatenate([dp, dr]), np.linalg.norm(dp), np.linalg.norm(dr)


def _clamp(dq, limit):
    """Clamp joint step magnitude."""
    return np.clip(dq, -limit, limit)


def _ik_dls(q, tpos, trot, qlo, qhi, maxiter, ptol, otol, damp, slim, use_urdf):
    """Damped Least Squares IK solver."""
    for it in range(maxiter):
        err, pe, oe = _pose_error(q, tpos, trot, use_urdf)
        if pe < ptol and oe < otol:
            return IKResult(q, True, pe, oe, it + 1, "Converged successfully.")
        J = compute_jacobian(q, use_urdf)
        dq = J.T @ np.linalg.solve(J @ J.T + (damp ** 2) * np.eye(6), err)
        q = np.clip(q + _clamp(dq, slim), qlo, qhi)
    err, pe, oe = _pose_error(q, tpos, trot, use_urdf)
    return IKResult(q, False, pe, oe, maxiter,
                    f"Did not converge after {maxiter} iters. "
                    f"pos_err={pe:.6f}, ori_err={oe:.6f}")


def _ik_nr(q, tpos, trot, qlo, qhi, maxiter, ptol, otol, slim, use_urdf):
    """Newton-Raphson IK solver."""
    for it in range(maxiter):
        err, pe, oe = _pose_error(q, tpos, trot, use_urdf)
        if pe < ptol and oe < otol:
            return IKResult(q, True, pe, oe, it + 1, "Converged successfully.")
        J = compute_jacobian(q, use_urdf)
        dq = np.linalg.pinv(J) @ err
        q = np.clip(q + _clamp(dq, slim), qlo, qhi)
    err, pe, oe = _pose_error(q, tpos, trot, use_urdf)
    return IKResult(q, False, pe, oe, maxiter,
                    f"Did not converge after {maxiter} iters. "
                    f"pos_err={pe:.6f}, ori_err={oe:.6f}")


def inverse_kinematics(
    target_pose,
    initial_guess=None,
    method='damped_least_squares',
    max_iterations=500,
    position_tolerance=1e-4,
    orientation_tolerance=1e-4,
    damping=0.1,
    step_limit=0.5,
    joint_limits_lower=None,
    joint_limits_upper=None,
    use_urdf=False,
):
    """
    Solve IK for the 6-DOF arm.

    Parameters
    ----------
    target_pose : (6,) or (4,4) -- desired end-effector pose.
    initial_guess : (6,) or None.
    method : 'damped_least_squares' or 'newton_raphson'.
    use_urdf : bool, if True use URDF-based FK.
    Returns IKResult named tuple.
    """
    target_pose = np.asarray(target_pose, dtype=float)
    if target_pose.shape == (4, 4):
        T_target = target_pose
    elif target_pose.shape == (6,):
        T_target = pose_to_transform(*target_pose)
    else:
        raise ValueError(f"target_pose must be (6,) or (4,4), got {target_pose.shape}")

    target_pos = T_target[:3, 3]
    target_rot = T_target[:3, :3]
    q = (np.zeros(NUM_JOINTS) if initial_guess is None
         else np.asarray(initial_guess, dtype=float).copy())
    q_lo = (JOINT_LIMITS_LOWER if joint_limits_lower is None
            else np.asarray(joint_limits_lower))
    q_hi = (JOINT_LIMITS_UPPER if joint_limits_upper is None
            else np.asarray(joint_limits_upper))

    if method == 'damped_least_squares':
        return _ik_dls(q, target_pos, target_rot, q_lo, q_hi,
                       max_iterations, position_tolerance, orientation_tolerance,
                       damping, step_limit, use_urdf)
    if method == 'newton_raphson':
        return _ik_nr(q, target_pos, target_rot, q_lo, q_hi,
                      max_iterations, position_tolerance, orientation_tolerance,
                      step_limit, use_urdf)
    raise ValueError(f"Unknown IK method: {method}")


# ==================== Rotation Utilities ====================

def rotation_matrix_to_euler(R):
    """Extract (roll, pitch, yaw) from R. Convention: R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Build R from (roll, pitch, yaw). R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ],
    ])


def pose_to_transform(x, y, z, roll, pitch, yaw):
    """Build 4x4 homogeneous transform from (x, y, z, roll, pitch, yaw)."""
    T = np.eye(4)
    T[:3, :3] = euler_to_rotation_matrix(roll, pitch, yaw)
    T[:3, 3] = [x, y, z]
    return T


# ==================== Tests ====================

def _sep(title=""):
    w = 72
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * pad)
    else:
        print("=" * w)


def _fk_reference(joint_angles):
    """Reference FK replicating the DH chain from Inv_Dyn_2.py exactly."""
    q = np.asarray(joint_angles, dtype=float)
    th = np.zeros(6); d = np.zeros(6)
    a = np.zeros(6); alp = np.zeros(6)
    th[0] = q[0];                         d[0] = 0.1367; a[0] = 0;      alp[0] = 0
    th[1] = q[1] + 173.01 / 180 * np.pi; d[1] = 0;      a[1] = 0;      alp[1] = np.pi / 2
    th[2] = q[2] - 157.51 / 180 * np.pi; d[2] = 0;      a[2] = 0.3846; alp[2] = 0
    th[3] = q[3] - 15.5 / 180 * np.pi;   d[3] = 0;      a[3] = 0.3179; alp[3] = 0
    th[4] = q[4] - np.pi / 2;            d[4] = 0;      a[4] = 0.0715; alp[4] = -np.pi / 2
    th[5] = q[5];                         d[5] = 0.026;  a[5] = 0;      alp[5] = -np.pi / 2
    T = np.eye(4)
    for i in range(6):
        T = T @ dh_transform(alp[i], a[i], d[i], th[i])
    return T


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True, linewidth=120)
    results = {}

    # --- Test 1: DH FK at zero angles ---
    _sep("Test 1: DH FK at zero joint angles")
    q0 = np.zeros(6)
    T0 = forward_kinematics(q0)
    Tref0 = _fk_reference(q0)
    m1 = np.allclose(T0, Tref0, atol=1e-10)
    print(f"Position: {T0[:3, 3]}")
    print(f"FK matches Inv_Dyn_2 reference: {m1}")
    results["DH FK ref (zero)"] = m1

    # --- Test 2: DH FK at non-zero angles ---
    _sep("Test 2: DH FK at non-zero joint angles")
    qt = np.array([0.3, -0.5, 0.8, -0.2, 0.6, -0.4])
    Tt = forward_kinematics(qt)
    Treft = _fk_reference(qt)
    m2 = np.allclose(Tt, Treft, atol=1e-10)
    print(f"Position: {Tt[:3, 3]}")
    print(f"FK matches Inv_Dyn_2 reference: {m2}")
    results["DH FK ref (nonzero)"] = m2

    # --- Test 3: DH Jacobian verification ---
    _sep("Test 3: DH Jacobian (analytical vs numerical)")
    Ja_dh = compute_jacobian(qt, use_urdf=False)
    Jn_dh = compute_jacobian_numerical(qt, use_urdf=False)
    jd_dh = np.max(np.abs(Ja_dh - Jn_dh))
    print(f"Max |analytical - numerical| = {jd_dh:.2e}")
    ok_jdh = jd_dh < 1e-4
    print(f"Match: {ok_jdh}")
    results["DH Jacobian"] = ok_jdh

    # --- Test 4: URDF FK at zero angles ---
    _sep("Test 4: URDF FK at zero joint angles")
    T0u = forward_kinematics(q0, use_urdf=True)
    print(f"URDF position (q=0): {T0u[:3, 3]}")
    print(f"DH   position (q=0): {T0[:3, 3]}")
    pos_diff = np.linalg.norm(T0u[:3, 3] - T0[:3, 3])
    print(f"Position difference DH vs URDF: {pos_diff:.4f} m")
    results["URDF FK runs"] = True

    # --- Test 5: URDF Jacobian verification ---
    _sep("Test 5: URDF Jacobian (analytical vs numerical)")
    Ja_urdf = compute_jacobian(qt, use_urdf=True)
    Jn_urdf = compute_jacobian_numerical(qt, use_urdf=True)
    jd_urdf = np.max(np.abs(Ja_urdf - Jn_urdf))
    print(f"Max |analytical - numerical| = {jd_urdf:.2e}")
    ok_jurdf = jd_urdf < 1e-4
    print(f"Match: {ok_jurdf}")
    results["URDF Jacobian"] = ok_jurdf

    # --- Test 6: DH IK round-trip (DLS) ---
    _sep("Test 6: DH IK round-trip (DLS)")
    q_orig = np.array([0.3, -0.5, 0.8, -0.2, 0.6, -0.4])
    Ttgt = forward_kinematics(q_orig)
    tgt = get_end_effector_pose(q_orig)
    ig = q_orig + np.random.RandomState(42).uniform(-0.2, 0.2, 6)
    r1 = inverse_kinematics(tgt, initial_guess=ig, method="damped_least_squares")
    Trec1 = forward_kinematics(r1.joint_angles)
    rte1 = np.linalg.norm(Ttgt[:3, 3] - Trec1[:3, 3])
    print(f"Converged: {r1.success}  iters: {r1.iterations}")
    print(f"pos_err={r1.position_error:.2e}  ori_err={r1.orientation_error:.2e}")
    print(f"Round-trip pos err: {rte1:.2e}")
    results["DH IK DLS"] = r1.success and rte1 < 1e-3

    # --- Test 7: DH IK round-trip (NR) ---
    _sep("Test 7: DH IK round-trip (Newton-Raphson)")
    r2 = inverse_kinematics(tgt, initial_guess=ig, method="newton_raphson")
    Trec2 = forward_kinematics(r2.joint_angles)
    rte2 = np.linalg.norm(Ttgt[:3, 3] - Trec2[:3, 3])
    print(f"Converged: {r2.success}  iters: {r2.iterations}")
    print(f"pos_err={r2.position_error:.2e}  ori_err={r2.orientation_error:.2e}")
    print(f"Round-trip pos err: {rte2:.2e}")
    results["DH IK NR"] = r2.success and rte2 < 1e-3

    # --- Test 8: URDF IK round-trip (DLS) ---
    _sep("Test 8: URDF IK round-trip (DLS)")
    Ttgt_u = forward_kinematics(q_orig, use_urdf=True)
    tgt_u = get_end_effector_pose(q_orig, use_urdf=True)
    ig_u = q_orig + np.random.RandomState(42).uniform(-0.2, 0.2, 6)
    r3 = inverse_kinematics(tgt_u, initial_guess=ig_u,
                            method="damped_least_squares", use_urdf=True)
    Trec3 = forward_kinematics(r3.joint_angles, use_urdf=True)
    rte3 = np.linalg.norm(Ttgt_u[:3, 3] - Trec3[:3, 3])
    print(f"Converged: {r3.success}  iters: {r3.iterations}")
    print(f"pos_err={r3.position_error:.2e}  ori_err={r3.orientation_error:.2e}")
    print(f"Round-trip pos err: {rte3:.2e}")
    results["URDF IK DLS"] = r3.success and rte3 < 1e-3

    # --- Test 9: DH IK from zero guess ---
    _sep("Test 9: DH IK from zero initial guess")
    r4 = inverse_kinematics(tgt, initial_guess=np.zeros(6),
                            method="damped_least_squares", max_iterations=1000)
    print(f"Converged: {r4.success}  iters: {r4.iterations}")
    print(f"pos_err={r4.position_error:.2e}  ori_err={r4.orientation_error:.2e}")
    results["DH IK zero-guess"] = r4.success

    # --- Summary ---
    _sep("Summary")
    all_pass = True
    for name, ok in results.items():
        tag = "PASS" if ok else "FAIL"
        print(f"  {name:30s} {tag}")
        if not ok:
            all_pass = False
    print()
    print("Overall:", "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
