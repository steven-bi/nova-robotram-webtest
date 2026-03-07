"""Pre-convert URDF to USD."""
import sys
sys.path.insert(0, "/root/autodl-tmp/arm_grasp")

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

cfg = UrdfConverterCfg(
    asset_path="/root/autodl-tmp/arm_grasp/arm/urdf/arm.urdf",
    usd_dir="/root/autodl-tmp/arm_grasp/arm/usd",
    usd_file_name="arm.usd",
    fix_base=True,
    merge_fixed_joints=False,
    force_usd_conversion=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    ),
)

converter = UrdfConverter(cfg)
usd_path = converter.usd_path
print(f"\nUSD generated at: {usd_path}")

import os
if os.path.exists(usd_path):
    size = os.path.getsize(usd_path)
    print(f"USD file size: {size} bytes")
    usd_dir = os.path.dirname(usd_path)
    for f in sorted(os.listdir(usd_dir)):
        fpath = os.path.join(usd_dir, f)
        print(f"  {f}: {os.path.getsize(fpath)} bytes")
else:
    print("ERROR: USD file not created!")

simulation_app.close()
