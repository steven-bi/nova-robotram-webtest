"""
Kinematics module for the 6-DOF robot arm.
Provides Forward Kinematics (FK), Inverse Kinematics (IK), and Jacobian computation.
Supports both DH-based and URDF-based FK modes.
"""

from .kinematics import (
    forward_kinematics,
    get_all_transforms,
    get_end_effector_position,
    get_end_effector_pose,
    compute_jacobian,
    inverse_kinematics,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    pose_to_transform,
    IKResult,
    NUM_JOINTS,
    DH_ALPHA, DH_A, DH_D, DH_THETA_OFFSET,
    JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER,
)
