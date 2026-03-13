
import arm_grasp
import os
print("ARM_GRASP_DATA_DIR:", arm_grasp.ARM_GRASP_DATA_DIR)
print("URDF exists:", os.path.exists(os.path.join(arm_grasp.ARM_GRASP_DATA_DIR, "urdf", "arm.urdf")))

from arm_grasp.assets import ARM_6DOF_CFG
print("ARM_6DOF_CFG loaded:", type(ARM_6DOF_CFG).__name__)
print("fix_base:", ARM_6DOF_CFG.spawn.fix_base)

from arm_grasp.agents import ArmLiftCubePPORunnerCfg
cfg = ArmLiftCubePPORunnerCfg()
print("PPO config loaded, max_iterations:", cfg.max_iterations)
print("experiment_name:", cfg.experiment_name)
print("All imports OK!")
