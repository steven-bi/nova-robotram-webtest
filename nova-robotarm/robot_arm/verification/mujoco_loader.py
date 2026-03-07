"""
MuJoCo model loading utility for the 6-DOF robot arm.

Loads the URDF (with STL meshes) into MuJoCo, fixing the ROS package:// paths.
Provides helper functions for FK/Jacobian queries via MuJoCo.
"""

import os
import numpy as np
import mujoco

# Default paths
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH = os.path.join(_BASE_DIR, "arm", "urdf", "arm.urdf")
MESHES_DIR = os.path.join(_BASE_DIR, "arm", "meshes")
MESH_FILES = [
    "0.STL", "1.STL", "2.STL", "3.STL", "4.STL", "5.STL", "6.STL",
    "7-1.STL", "7-2.STL",
]

# The MuJoCo body corresponding to our URDF FK end-effector (with gripper offset)
EE_BODY_NAME = "7-1"


def load_mujoco_model(urdf_path=None, meshes_dir=None):
    """
    Load the robot URDF into MuJoCo.

    Replaces package://arm/meshes/ URIs with bare filenames and passes
    the STL mesh files as an in-memory asset dictionary.

    Returns (mujoco.MjModel, mujoco.MjData).
    """
    urdf_path = urdf_path or URDF_PATH
    meshes_dir = meshes_dir or MESHES_DIR

    with open(urdf_path, "r", encoding="utf-8") as f:
        urdf_str = f.read()

    # Strip ROS package URI prefix so MuJoCo sees bare filenames
    urdf_fixed = urdf_str.replace("package://arm/meshes/", "")

    # Load mesh files as binary bytes
    mesh_assets = {}
    for name in MESH_FILES:
        filepath = os.path.join(meshes_dir, name)
        with open(filepath, "rb") as f:
            mesh_assets[name] = f.read()

    model = mujoco.MjModel.from_xml_string(urdf_fixed, mesh_assets)
    data = mujoco.MjData(model)
    return model, data


def get_ee_body_id(model):
    """Return the MuJoCo body ID for the end-effector (gripper body 7-1)."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME)


def set_joint_angles(model, data, q6):
    """Set the 6 revolute joint angles and run mj_forward. Gripper stays at 0."""
    q6 = np.asarray(q6, dtype=float)
    data.qpos[:6] = q6
    data.qpos[6:] = 0.0
    mujoco.mj_forward(model, data)


def get_ee_pose(model, data):
    """
    Return (pos, rotmat) of the end-effector body after mj_forward.
    pos: (3,) position vector
    rotmat: (3,3) rotation matrix
    """
    bid = get_ee_body_id(model)
    pos = data.xpos[bid].copy()
    rotmat = data.xmat[bid].reshape(3, 3).copy()
    return pos, rotmat


def get_mujoco_jacobian(model, data):
    """
    Return (6, 6) geometric Jacobian at the end-effector body origin.
    Rows 0:3 = linear velocity, rows 3:6 = angular velocity.
    Only the first 6 columns (revolute joints) are returned.
    """
    bid = get_ee_body_id(model)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, data.xpos[bid], bid)
    # Stack and keep only revolute joint columns
    return np.vstack([jacp[:, :6], jacr[:, :6]])


def print_model_info(model):
    """Print basic model info for debugging."""
    print(f"MuJoCo model loaded: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
    print("Bodies:")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "(unnamed)"
        print(f"  [{i}] {name}")
    print("Joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "(unnamed)"
        jtype = model.jnt_type[i]
        type_str = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}[jtype]
        print(f"  [{i}] {name} type={type_str} qposadr={model.jnt_qposadr[i]}")
