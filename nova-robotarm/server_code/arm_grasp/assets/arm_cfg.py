"""Configuration for custom 6-DOF robot arm with parallel-jaw gripper."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from arm_grasp import ARM_GRASP_DATA_DIR

ARM_6DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        asset_path=f"{ARM_GRASP_DATA_DIR}/urdf/arm.urdf",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None, damping=None
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": 0.0, "joint_2": 0.0, "joint_3": 0.0,
            "joint_4": 0.0, "joint_5": 0.0, "joint_6": 0.0,
            "gripper_left_joint": 0.04, "gripper_right_joint": 0.04,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint_2"],
            effort_limit=300.0,
            velocity_limit=17.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["joint_3"],
            effort_limit=300.0,
            velocity_limit=17.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint_4", "joint_5", "joint_6"],
            effort_limit=100.0,
            velocity_limit=17.0,
            stiffness=400.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_left_joint", "gripper_right_joint"],
            effort_limit=20.0,
            velocity_limit=0.5,
            stiffness=200.0,
            damping=5.0,
        ),
    },
)
