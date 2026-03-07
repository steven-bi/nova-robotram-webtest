"""
Web-based Joint Control GUI with 3D URDF viewer.

Run:  python joint_gui_web.py
Open:  http://localhost:8888

Features:
  - 3D robot arm model (URDF + STL meshes) rendered with three.js
  - 6 joint sliders — drag to preview pose in 3D
  - "Read" syncs sliders to real motor positions
  - "Execute" smoothly moves real robot to target
  - "STOP" emergency stop
"""
import http.server
import json
import math
import os
import sys
import threading
import time
import webbrowser
from urllib.parse import urlparse
import numpy as np

# Add grpc_stream to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robot_arm', 'grpc_stream'))
from robstride_pcan import Motor, MotorControl

PORT = 8888
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Motor setup ──────────────────────────────────────────────────

MOTOR_CONFIG = [
    (1, '04', True,  15, 0.5),
    (2, '04', False, 15, 0.5),
    (3, '03', True,  15, 0.5),
    (4, '00', False, 80, 1.5),
    (5, '00', False, 50, 1.2),
    (6, '00', False, 50, 1.2),
]

# Safe home = natural gravity rest position (degrees → radians)
HOME_DEG = {1: 0.1, 2: 0.0, 3: 0.0, 4: -11.1, 5: -1.4, 6: 0.2}
HOME_RAD = {mid: deg * math.pi / 180 for mid, deg in HOME_DEG.items()}

# Joint limits (degrees) — clamp targets before sending to motors
JOINT_LIMITS_DEG = {
    1: (-150, 150),
    2: (-120, 120),
    3: (-120, 120),
    4: ( -90,  90),
    5: ( -90,  90),
    6: (-180, 180),
}

# ── Gravity compensation (from URDF) ─────────────────────────────
_G = np.array([0.0, 0.0, -9.81])  # Z-up world frame
GRAV_SCALE = 1.0  # global gravity feedforward scale
GRAV_SCALE_JOINT = {}  # temporarily cleared to diagnose startup protection trip

_JOINT_ORIGIN = [  # (xyz, rpy) fixed origin per joint, order = J1..J6
    ([0,       0,       0.1281 ], [0,       0,       0      ]),
    ([0,       0,       0      ], [-1.5708, 0,       3.1397 ]),
    ([0.37425, 0,       0      ], [0,       0,      -2.8166 ]),
    ([0.30395, 0.00033, 0      ], [0,       0,      -0.3307 ]),
    ([0.080376,0.0018, -0.002  ], [-1.5708,-1.5689,  0.0057 ]),
    ([0.002092,0.02993, 0      ], [-1.5708, 0,      -0.0019 ]),
]
_LINK_MASS = [1.177, 1.315, 0.86, 0.407, 0.35, 0.362]   # link 1-6 (link6 incl. gripper)
_LINK_COM  = [
    [-0.00051492, -0.00324,    -0.005     ],  # link 1
    [ 0.276,      -0.0116,      0         ],  # link 2
    [ 0.204,       0.0195,      0.00036064],  # link 3
    [ 0.0675,      0.0574,     -0.0038986 ],  # link 4
    [ 0.00043477,  0.0038,      0.0027    ],  # link 5
    [-0.001629,   -1.2741e-6,   0.044     ],  # link 6
]

def _rpy_mat(rpy):
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    return Rz @ Ry @ Rx

def compute_gravity_torques(q_dict):
    """Forward kinematics → gravity torques (Nm) for each joint.
    q_dict: {motor_id: angle_rad} in URDF convention (state_q, inverse already applied).
    """
    T = np.eye(4)
    j_pos, j_axis, l_com = [], [], []
    for i, (xyz, rpy) in enumerate(_JOINT_ORIGIN):
        q = q_dict.get(i + 1, 0.0)
        To = np.eye(4)
        To[:3, :3] = _rpy_mat(rpy)
        To[:3, 3]  = xyz
        T = T @ To
        j_pos.append(T[:3, 3].copy())
        j_axis.append(T[:3, 2].copy())          # joint Z-axis in world
        cq, sq = math.cos(q), math.sin(q)
        Tq = np.eye(4)
        Tq[:3, :3] = np.array([[cq,-sq,0],[sq,cq,0],[0,0,1]])
        T = T @ Tq
        com_h = np.append(_LINK_COM[i], 1.0)
        l_com.append((T @ com_h)[:3])

    # τ_i = z_i · Σ_{j≥i}  m_j * (p_com_j − p_joint_i) × g
    result = {}
    for i in range(6):
        tau = 0.0
        for j in range(i, 6):
            r = l_com[j] - j_pos[i]
            tau += float(np.dot(j_axis[i], np.cross(r, _LINK_MASS[j] * _G)))
        scale = GRAV_SCALE_JOINT.get(i + 1, GRAV_SCALE)
        result[i + 1] = -tau * scale
    return result

comm = None
motors = {}
motor_kp = {}
motor_kd = {}
motor_enabled = {}
moving = False
stop_flag = False
hold_pos = {}       # current hold positions (radians), updated after each move
_hold_thread = None
_can_lock = threading.Lock()

def _hold_loop():
    """Background thread: continuously send MIT hold commands when not moving."""
    while True:
        time.sleep(0.04)  # 25 Hz — faster hold reduces pre-motion sag window
        if moving or not hold_pos or comm is None:
            continue
        grav = compute_gravity_torques(dict(hold_pos))
        for mid, q in list(hold_pos.items()):
            if moving:
                break
            if motor_enabled.get(mid, False):
                try:
                    with _can_lock:
                        comm.MIT(mid, angle=q, velocity=0, torque=grav[mid],
                                 kp=motor_kp[mid], kd=motor_kd[mid])
                except Exception:
                    pass

def init_hardware():
    global comm
    comm = MotorControl()
    comm.opon_device()
    comm.clear_buffer()
    for mid, mtype, inv, kp, kd in MOTOR_CONFIG:
        m = Motor(mtype, mid, inv)
        comm.addMotor(m)
        motors[mid] = m
        motor_kp[mid] = kp
        motor_kd[mid] = kd
    for _ in range(3):
        for mid in motors:
            comm.Enable(mid)
        time.sleep(0.03)
    for mid in motors:
        motor_enabled[mid] = True
        hold_pos[mid] = motors[mid].state_q

    global _hold_thread
    _hold_thread = threading.Thread(target=_hold_loop, daemon=True)
    _hold_thread.start()

def shutdown_hardware():
    if comm:
        for mid in motors:
            try: comm.Disable(mid)
            except: pass
        try: comm.close_device()
        except: pass

def read_positions():
    # hold_loop maintains MIT state continuously; no Enable() needed
    return {mid: m.state_q for mid, m in motors.items()}

def execute_move(targets_rad):
    global moving, stop_flag
    if moving:
        return False
    stop_flag = False

    # Clamp all targets to joint limits
    for mid in list(targets_rad.keys()):
        lo, hi = JOINT_LIMITS_DEG[mid]
        lo_rad = lo * math.pi / 180
        hi_rad = hi * math.pi / 180
        clamped = max(lo_rad, min(hi_rad, targets_rad[mid]))
        if clamped != targets_rad[mid]:
            print(f"J{mid} target clamped: {targets_rad[mid]*180/math.pi:.1f}° → {clamped*180/math.pi:.1f}°")
        targets_rad[mid] = clamped

    # Use last known positions
    start = {mid: m.state_q for mid, m in motors.items()}

    # Send one hold frame WITH gravity compensation before stopping the hold loop
    grav_start = compute_gravity_torques(start)
    for mid in motors:
        if motor_enabled.get(mid, False):
            try:
                with _can_lock:
                    comm.MIT(mid, angle=start[mid], velocity=0, torque=grav_start[mid],
                             kp=motor_kp[mid], kd=motor_kd[mid])
            except Exception:
                pass

    moving = True   # now stop hold loop — motors already have fresh hold frame
    stop_flag = False

    # Safety: clamp start positions too (reject corrupt state_q)
    for mid in start:
        lo, hi = JOINT_LIMITS_DEG[mid]
        lo_rad = lo * math.pi / 180
        hi_rad = hi * math.pi / 180
        if not (lo_rad <= start[mid] <= hi_rad):
            print(f"J{mid} start out of limits ({start[mid]*180/math.pi:.1f}°), clamping to target")
            start[mid] = targets_rad[mid]

    duration = 2.0
    dt = 0.01
    t_start = time.time()

    try:
        while not stop_flag:
            t = time.time() - t_start
            if t >= duration:
                t = duration
            alpha = 0.5 * (1 - math.cos(math.pi * t / duration))
            q_now = {mid: m.state_q for mid, m in motors.items()}
            grav = compute_gravity_torques(q_now)
            for mid in motors:
                q_des = start[mid] + alpha * (targets_rad[mid] - start[mid])
                try:
                    with _can_lock:
                        comm.MIT(mid, angle=q_des, velocity=0, torque=grav[mid],
                                 kp=motor_kp[mid], kd=motor_kd[mid])
                except Exception as can_err:
                    print(f"CAN error J{mid}: {can_err}")
            if t >= duration:
                break
            time.sleep(dt)
    except Exception as e:
        print(f"Move error: {e}")
    finally:
        # Settle phase: keep sending MIT at target for 0.5s to let motor converge past friction
        if not stop_flag:
            settle_end = time.time() + 0.5
            grav_final = compute_gravity_torques(targets_rad)
            while time.time() < settle_end:
                for mid in motors:
                    if motor_enabled.get(mid, False):
                        try:
                            with _can_lock:
                                comm.MIT(mid, angle=targets_rad[mid], velocity=0,
                                         torque=grav_final[mid],
                                         kp=motor_kp[mid], kd=motor_kd[mid])
                        except Exception:
                            pass
                time.sleep(0.02)  # 50 Hz settle rate
        # Hold at target positions so motor continues converging after trajectory ends
        for mid in motors:
            hold_pos[mid] = targets_rad[mid]
        moving = False
    return True

def emergency_stop():
    global stop_flag, moving
    stop_flag = True
    for mid in motors:
        try: comm.Disable(mid)
        except: pass
        motor_enabled[mid] = False
    moving = False

# ── HTTP handler ─────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress logs

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _json_response(self, data, code=200):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _serve_file(self, filepath, content_type):
        if not os.path.isfile(filepath):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self._cors()
        self.end_headers()
        with open(filepath, 'rb') as f:
            self.wfile.write(f.read())

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self._serve_file(os.path.join(BASE_DIR, 'web', 'index.html'), 'text/html')
        elif path.startswith('/meshes/'):
            stl = os.path.join(BASE_DIR, 'arm', 'meshes', os.path.basename(path))
            self._serve_file(stl, 'application/octet-stream')
        elif path == '/urdf/arm.urdf':
            self._serve_file(os.path.join(BASE_DIR, 'arm', 'urdf', 'arm.urdf'),
                             'application/xml')
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''

        if path == '/api/read':
            pos = read_positions()
            self._json_response({str(k): v for k, v in pos.items()})

        elif path == '/api/execute':
            data = json.loads(body)
            targets = {int(k): float(v) for k, v in data.items()}
            thread = threading.Thread(target=execute_move, args=(targets,), daemon=True)
            thread.start()
            self._json_response({'ok': True})

        elif path == '/api/stop':
            emergency_stop()
            self._json_response({'ok': True})

        elif path == '/api/enable':
            for _ in range(3):
                for mid in motors:
                    comm.Enable(mid)
                time.sleep(0.03)
            for mid in motors:
                motor_enabled[mid] = True
            self._json_response({'ok': True})

        elif path == '/api/enable_motor':
            data = json.loads(body)
            mid = int(data['id'])
            for _ in range(3):
                comm.Enable(mid)
                time.sleep(0.01)
            motor_enabled[mid] = True
            self._json_response({'ok': True, 'id': mid, 'enabled': True})

        elif path == '/api/disable_motor':
            data = json.loads(body)
            mid = int(data['id'])
            comm.Disable(mid)
            motor_enabled[mid] = False
            self._json_response({'ok': True, 'id': mid, 'enabled': False})

        elif path == '/api/motor_states':
            self._json_response({str(k): v for k, v in motor_enabled.items()})

        elif path == '/api/set_zero':
            data = json.loads(body)
            mid = int(data['id'])
            comm.Set_Zero(mid)
            self._json_response({'ok': True, 'id': mid})

        elif path == '/api/home':
            thread = threading.Thread(target=execute_move, args=(HOME_RAD,), daemon=True)
            thread.start()
            self._json_response({'ok': True, 'targets': {str(k): round(v*180/math.pi,1) for k,v in HOME_RAD.items()}})

        else:
            self.send_error(404)

# ── Main ─────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(BASE_DIR, 'web'), exist_ok=True)

    print("Initializing PCAN hardware...")
    init_hardware()
    print("Hardware ready.\n")

    server = http.server.HTTPServer(('0.0.0.0', PORT), Handler)
    print(f"Server running at http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")

    webbrowser.open(f'http://localhost:{PORT}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        shutdown_hardware()
        server.server_close()

if __name__ == '__main__':
    main()
