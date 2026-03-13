"""
workspace_analysis.py — Monte Carlo workspace visualization for Nova robot arm.

Usage:
    python workspace_analysis.py

Output:
    workspace.html  — interactive 3D point cloud (open in browser)

No hardware required. Pure forward kinematics + random sampling.
"""
import sys, os
import math
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot_arm.kinematics.kinematics import forward_kinematics

# ── Joint limits (calibrated, degrees) ───────────────────────────
JOINT_LIMITS_DEG = {
    1: (-175, 175),
    2: ( -51,   0),
    3: (   0, 170),
    4: ( -10,  90),
    5: (-113, 113),
    6: (-180, 180),
}

N_SAMPLES = 150_000   # increase for denser cloud (costs more time)
SEED      = 42

# ── Sampling ──────────────────────────────────────────────────────
print(f"Sampling {N_SAMPLES:,} random configurations...")
rng = np.random.default_rng(SEED)

lo = np.array([JOINT_LIMITS_DEG[i+1][0] for i in range(6)]) * math.pi / 180
hi = np.array([JOINT_LIMITS_DEG[i+1][1] for i in range(6)]) * math.pi / 180

q_samples = rng.uniform(lo, hi, size=(N_SAMPLES, 6))

xs, ys, zs = [], [], []
for i, q in enumerate(q_samples):
    if i % 10000 == 0:
        print(f"  {i:>7,} / {N_SAMPLES:,}", end='\r')
    T = forward_kinematics(q, use_urdf=True)
    xs.append(float(T[0, 3]) * 1000)   # mm
    ys.append(float(T[1, 3]) * 1000)
    zs.append(float(T[2, 3]) * 1000)

xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)
print(f"\nDone. {len(xs):,} points computed.")

# ── Statistics ───────────────────────────────────────────────────
r_xy = np.sqrt(xs**2 + ys**2)  # horizontal reach
r_3d = np.sqrt(xs**2 + ys**2 + zs**2)

print(f"\n── Workspace statistics ──────────────────────")
print(f"X:           {xs.min():.0f} ~ {xs.max():.0f} mm")
print(f"Y:           {ys.min():.0f} ~ {ys.max():.0f} mm")
print(f"Z:           {zs.min():.0f} ~ {zs.max():.0f} mm")
print(f"Horiz reach: {r_xy.min():.0f} ~ {r_xy.max():.0f} mm")
print(f"3D reach:    {r_3d.min():.0f} ~ {r_3d.max():.0f} mm")

# ── Color by Z height ─────────────────────────────────────────────
# Normalize Z to [0,1] for color mapping
z_norm = (zs - zs.min()) / (zs.max() - zs.min() + 1e-9)

# ── HTML output with Plotly (no pip install needed — CDN) ─────────
print("\nGenerating workspace.html ...")

# Downsample for browser performance (50k points renders well)
DISPLAY_N = min(50_000, len(xs))
idx = rng.choice(len(xs), DISPLAY_N, replace=False)
dx, dy, dz    = xs[idx], ys[idx], zs[idx]
dz_norm       = z_norm[idx]

# Serialize data as proper JSON
jx      = json.dumps(dx.round(1).tolist())
jy      = json.dumps(dy.round(1).tolist())
jz      = json.dumps(dz.round(1).tolist())
jcolor  = json.dumps(dz_norm.round(3).tolist())

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Nova Robot Arm — Workspace Analysis</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:'Segoe UI',Arial,sans-serif; }}
  #header {{ padding:16px 24px 8px; }}
  h1 {{ margin:0; font-size:20px; color:#00d2ff; }}
  #stats {{ margin:8px 0 0; font-size:13px; color:#aaa; line-height:1.8; }}
  #plot {{ width:100vw; height:calc(100vh - 110px); }}
</style>
</head>
<body>
<div id="header">
  <h1>Nova Robot Arm — Reachable Workspace</h1>
  <div id="stats">
    Samples: <b style="color:#fff">{N_SAMPLES:,}</b> &nbsp;|&nbsp;
    Display: <b style="color:#fff">{DISPLAY_N:,}</b> &nbsp;|&nbsp;
    X: <b style="color:#00d2ff">{xs.min():.0f} ~ {xs.max():.0f} mm</b> &nbsp;|&nbsp;
    Y: <b style="color:#00d2ff">{ys.min():.0f} ~ {ys.max():.0f} mm</b> &nbsp;|&nbsp;
    Z: <b style="color:#00d2ff">{zs.min():.0f} ~ {zs.max():.0f} mm</b> &nbsp;|&nbsp;
    Max reach: <b style="color:#ffd700">{r_3d.max():.0f} mm</b>
    &nbsp;&nbsp;|&nbsp; Joint limits — J1:±175° J2:−51~0° J3:0~170° J4:−10~90° J5:±113° J6:±180°
  </div>
</div>
<div id="plot"></div>
<script>
var trace = {{
  type: 'scatter3d',
  mode: 'markers',
  x: {jx},
  y: {jy},
  z: {jz},
  marker: {{
    size: 1.5,
    color: {jcolor},
    colorscale: 'Jet',
    opacity: 0.5,
    showscale: true,
    colorbar: {{ title: 'Z (normalized)', thickness: 14, len: 0.7 }}
  }},
  hovertemplate: 'X: %{{x:.0f}}mm<br>Y: %{{y:.0f}}mm<br>Z: %{{z:.0f}}mm<extra></extra>'
}};

var origin = {{
  type: 'scatter3d', mode: 'markers',
  x:[0], y:[0], z:[0],
  marker:{{size:10, color:'#ff4444', symbol:'cross'}},
  name:'Base'
}};

var layout = {{
  paper_bgcolor: '#111',
  plot_bgcolor:  '#111',
  scene: {{
    bgcolor: '#111',
    xaxis: {{ title:'X (mm)', color:'#aaa', gridcolor:'#333' }},
    yaxis: {{ title:'Y (mm)', color:'#aaa', gridcolor:'#333' }},
    zaxis: {{ title:'Z (mm)', color:'#aaa', gridcolor:'#333' }},
    aspectmode: 'data',
    camera: {{ eye: {{x:1.5, y:1.5, z:0.8}} }}
  }},
  margin: {{l:0,r:0,t:0,b:0}},
  legend: {{font:{{color:'#eee'}}}}
}};

Plotly.newPlot('plot', [trace, origin], layout, {{responsive:true}});
</script>
</body>
</html>"""

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'workspace.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Saved: {out_path}")
print("Open workspace.html in your browser to view the 3D workspace.")
