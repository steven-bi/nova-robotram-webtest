import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaacsim.core.utils.stage import add_reference_to_stage

cfg = UrdfConverterCfg(
    asset_path="/root/gpufree-data/nova_training/assets/arm.urdf",
    usd_dir="/root/gpufree-data/nova_training/assets/",
    usd_file_name="arm.usd",
    fix_base=True,
    merge_fixed_joints=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=100.0,
            damping=10.0,
        ),
    ),
)
print("Converting URDF to USD...")
converter = UrdfConverter(cfg)
print(f"USD saved to: {converter.usd_path}")

sim_cfg = sim_utils.SimulationCfg(dt=0.01)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])

# 深灰背景穹顶光
sim_utils.DomeLightCfg(
    intensity=500.0,
    color=(0.2, 0.2, 0.2),
).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=500.0, color=(0.2, 0.2, 0.2)))

# 正面主光源
sim_utils.DistantLightCfg(
    intensity=3000.0,
    color=(1.0, 1.0, 1.0),
).func("/World/MainLight", sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)))

# 深色地面
sim_utils.GroundPlaneCfg(
    color=(0.1, 0.1, 0.1),
).func("/World/GroundPlane", sim_utils.GroundPlaneCfg(color=(0.1, 0.1, 0.1)))

# 加载机械臂
add_reference_to_stage(converter.usd_path, "/World/Robot")

sim.reset()
print("Robot loaded! Running simulation...")
for i in range(500):
    sim.step()

print("Validation complete.")
simulation_app.close()
