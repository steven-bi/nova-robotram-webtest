"""Visualize trained policy trajectory as 3D animation."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Load trajectory data
with open("D:/inovxio/products/robotarm_ws/trajectory_data.json") as f:
    data = json.load(f)

steps = [d["step"] for d in data]
ee_x = [d["ee_x"] for d in data]
ee_y = [d["ee_y"] for d in data]
ee_z = [d["ee_z"] for d in data]
obj_x = [d["obj_x"] for d in data]
obj_y = [d["obj_y"] for d in data]
obj_z = [d["obj_z"] for d in data]
dists = [d["distance"] for d in data]
rewards = [d["reward"] for d in data]

# Convert to robot-local coordinates (subtract env offset for env 0)
# The env positions are in world coords with spacing

# Create figure with subplots
fig = plt.figure(figsize=(18, 10))

# 1. 3D trajectory
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot(ee_x, ee_y, ee_z, 'b-', alpha=0.6, linewidth=1, label='EE path')
ax1.plot(obj_x, obj_y, obj_z, 'r-', alpha=0.6, linewidth=1, label='Object path')
ax1.scatter(ee_x[0], ee_y[0], ee_z[0], c='blue', s=100, marker='^', label='EE start')
ax1.scatter(obj_x[0], obj_y[0], obj_z[0], c='red', s=100, marker='s', label='Obj start')
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_zlabel('Z (m)')
ax1.set_title('3D Trajectory')
ax1.legend(fontsize=8)

# 2. Distance over time
ax2 = fig.add_subplot(222)
ax2.plot(steps, dists, 'g-', linewidth=1)
ax2.axhline(y=0.08, color='r', linestyle='--', label='Grasp threshold (8cm)')
ax2.set_xlabel('Step')
ax2.set_ylabel('EE-Object Distance (m)')
ax2.set_title('EE-Object Distance Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Object height over time
ax3 = fig.add_subplot(223)
ax3.plot(steps, obj_z, 'r-', linewidth=1, label='Object height')
ax3.axhline(y=0.77, color='gray', linestyle='--', label='Initial height (0.77m)')
ax3.axhline(y=0.82, color='orange', linestyle='--', label='Lift threshold (0.82m)')
ax3.set_xlabel('Step')
ax3.set_ylabel('Object Z (m)')
ax3.set_title('Object Height Over Time')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Reward over time
ax4 = fig.add_subplot(224)
cumrew = np.cumsum(rewards)
ax4.plot(steps, cumrew, 'purple', linewidth=1)
ax4.set_xlabel('Step')
ax4.set_ylabel('Cumulative Reward')
ax4.set_title('Cumulative Reward')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/inovxio/products/robotarm_ws/training_result.png', dpi=150, bbox_inches='tight')
print("Saved training_result.png")

# Create animated GIF
fig2, (ax_3d, ax_dist) = plt.subplots(1, 2, figsize=(14, 6),
                                        subplot_kw={'projection': '3d'} if False else {})
fig2.clf()
ax_3d = fig2.add_subplot(121, projection='3d')
ax_dist = fig2.add_subplot(122)

trail_len = 30  # Show last 30 steps of trail

def update(frame):
    ax_3d.cla()
    ax_dist.cla()

    i = frame * 2  # Skip every other frame
    if i >= len(data):
        i = len(data) - 1

    # 3D view
    start = max(0, i - trail_len)
    ax_3d.plot(ee_x[start:i+1], ee_y[start:i+1], ee_z[start:i+1], 'b-', alpha=0.5, linewidth=2)
    ax_3d.plot(obj_x[start:i+1], obj_y[start:i+1], obj_z[start:i+1], 'r-', alpha=0.5, linewidth=2)
    ax_3d.scatter([ee_x[i]], [ee_y[i]], [ee_z[i]], c='blue', s=80, marker='o', label='EE')
    ax_3d.scatter([obj_x[i]], [obj_y[i]], [obj_z[i]], c='red', s=80, marker='s', label='Cube')

    # Set consistent axis limits
    all_x = ee_x + obj_x
    all_y = ee_y + obj_y
    all_z = ee_z + obj_z
    cx, cy = np.mean(all_x), np.mean(all_y)
    span = 0.8
    ax_3d.set_xlim(cx - span, cx + span)
    ax_3d.set_ylim(cy - span, cy + span)
    ax_3d.set_zlim(0.5, 1.5)
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title(f'Step {i} | Dist: {dists[i]:.3f}m')
    ax_3d.legend(fontsize=8)

    # Distance plot
    ax_dist.plot(steps[:i+1], dists[:i+1], 'g-', linewidth=1.5)
    ax_dist.axhline(y=0.08, color='r', linestyle='--', alpha=0.5, label='Grasp zone')
    ax_dist.set_xlim(0, 400)
    ax_dist.set_ylim(0, 0.5)
    ax_dist.set_xlabel('Step')
    ax_dist.set_ylabel('Distance (m)')
    ax_dist.set_title('EE-Object Distance')
    ax_dist.grid(True, alpha=0.3)

    return []

n_frames = len(data) // 2
anim = FuncAnimation(fig2, update, frames=n_frames, interval=50, blit=False)
anim.save('D:/inovxio/products/robotarm_ws/policy_animation.gif', writer='pillow', fps=20)
print("Saved policy_animation.gif")

plt.show()
