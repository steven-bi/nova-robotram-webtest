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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robstride_pcan import Motor, MotorControl
from robot_arm.kinematics.kinematics import (
    forward_kinematics, rotation_matrix_to_euler,
    inverse_kinematics, euler_to_rotation_matrix,
)

PORT = 8888
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Motor setup ──────────────────────────────────────────────────

MOTOR_CONFIG = [
    (1, '04', True,  15, 0.5),
    (2, '04', False, 15, 0.3),
    (3, '03', True,  15, 0.2),
    (4, '00', False, 80, 1.5),
    (5, '00', False, 50, 1.2),
    (6, '00', False, 50, 1.2),
    (7, '00', False, 10, 3.0),  # J7 Gripper: kd=3.0 to damp spring-assisted closing
]

# Safe home = natural gravity rest position (degrees → radians)
HOME_DEG = {1: 0.1, 2: 0.0, 3: 0.0, 4: -11.1, 5: -1.4, 6: 0.2, 7: 90.0}  # J7: open on home
HOME_RAD = {mid: deg * math.pi / 180 for mid, deg in HOME_DEG.items()}

# Joint limits (degrees) — clamp targets before sending to motors
JOINT_LIMITS_DEG = {
    1: (-175, 175),   # 360°可转，限制防缠线
    2: ( -51,   0),   # 0°=自然垂落(机械止挡)，-51°=最大上抬；J3大角度时物理可达-180°，自碰撞层处理
    3: (   0, 170),   # 0°=垂落，170°=最大伸展；防触地留余量
    4: ( -10,  90),   # 保守值；J3≥68.6°时可扩展至-120°，留给路径规划
    5: (-113, 113),
    6: (-180, 180),   # 360°可转，限制防缠线
    7: (   0,  90),   # Gripper: 0°=closed, 90°=fully open
}

# ── Gravity compensation (from URDF) ─────────────────────────────
_G = np.array([0.0, 0.0, -9.81])  # Z-up world frame
GRAV_SCALE = 1.0  # global gravity feedforward scale
GRAV_SCALE_JOINT = {2: 1.5}  # J2: confirmed fix for high-angle holding noise

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

# Gripper ramp state — updated by hold_loop, no separate thread needed
_gripper_ramp_target = None   # target rad; None = idle
_gripper_torque_limit = None  # Nm; stop closing when |tau| >= this
_gripper_lock = threading.Lock()

_gripper_done    = threading.Event()   # set when gripper ramp completes
_gripper_grasped = False               # True if torque limit triggered (object detected)

_sequence_state      = {'active': False, 'step': 0, 'total_steps': 8, 'step_name': 'idle', 'error': None}
_sequence_stop_flag  = False

# ── IK ────────────────────────────────────────────────────────────
_ik_preview_result = None   # {mid: angle_rad} from last successful preview
_ik_lock = threading.Lock()

# Joint limits in radians for IK solver (must match JOINT_LIMITS_DEG)
_IK_Q_LO = np.array([JOINT_LIMITS_DEG[i+1][0] * math.pi / 180 for i in range(6)])
_IK_Q_HI = np.array([JOINT_LIMITS_DEG[i+1][1] * math.pi / 180 for i in range(6)])

def _ik_target_transform(x_mm, y_mm, z_mm, orientation):
    """Build 4x4 target transform from xyz (mm) and orientation preset."""
    x, y, z = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
    if orientation == 'vertical':
        R = euler_to_rotation_matrix(np.pi, 0.0, 0.0)   # flip 180° about X → gripper down
    else:  # 'horizontal' (default): gripper pointing +X, Roll=-90°, Pitch=0°, Yaw=-90°
        R = euler_to_rotation_matrix(-np.pi/2, 0.0, -np.pi/2)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

# ── Teach / Drag recording ────────────────────────────────────────
TEACH_FILE = os.path.join(BASE_DIR, 'teachings.json')
MAX_TEACHINGS = 5
teach_mode = False
teach_recording = []
_teach_thread = None
teachings = []
teach_play_state = {'phase': 'idle', 'progress': 0, 'name': ''}

def _load_teachings():
    global teachings
    if os.path.isfile(TEACH_FILE):
        try:
            with open(TEACH_FILE) as f:
                teachings = json.load(f)
        except Exception:
            teachings = []

def _save_teachings():
    with open(TEACH_FILE, 'w') as f:
        json.dump(teachings, f)

def _smooth_frames(frames, window=5):
    """Moving average over joint positions to reduce hand tremor."""
    if len(frames) <= window:
        return frames
    mids = list(frames[0]['q'].keys())
    half = window // 2
    smoothed = []
    for i in range(len(frames)):
        lo = max(0, i - half)
        hi = min(len(frames), i + half + 1)
        avg_q = {mid: sum(frames[j]['q'][mid] for j in range(lo, hi)) / (hi - lo)
                 for mid in mids}
        smoothed.append({'t': frames[i]['t'], 'q': avg_q})
    return smoothed

def _teach_loop():
    global teach_recording
    teach_recording = []
    t0 = time.time()
    while teach_mode:
        t = time.time() - t0
        q = {str(mid): m.state_q for mid, m in motors.items()}
        teach_recording.append({'t': round(t, 3), 'q': q})
        time.sleep(0.05)  # 20 Hz

def teach_start():
    global teach_mode, _teach_thread
    if teach_mode or moving:
        return False
    teach_mode = True
    _teach_thread = threading.Thread(target=_teach_loop, daemon=True)
    _teach_thread.start()
    return True

def teach_stop():
    global teach_mode, teachings
    for mid in motors:
        hold_pos[mid] = motors[mid].state_q
    teach_mode = False
    time.sleep(0.12)  # let teach_loop exit
    frames = _smooth_frames(list(teach_recording))
    if len(frames) < 10:
        return None  # too short
    duration = round(frames[-1]['t'], 1)
    idx = len(teachings) + 1
    name = f'示教动作{idx}'
    entry = {'name': name, 'frames': frames, 'duration': duration, 'count': len(frames)}
    teachings.append(entry)
    if len(teachings) > MAX_TEACHINGS:
        teachings.pop(0)
    _save_teachings()
    return entry

def play_teaching(frames, name=''):
    global moving, stop_flag, teach_play_state
    if moving:
        return False
    # Step 1: smoothly move to first frame position
    teach_play_state = {'phase': 'moving', 'progress': 0, 'name': name}
    first_q = {int(mid): val for mid, val in frames[0]['q'].items()}
    execute_move(first_q)
    if stop_flag:
        teach_play_state = {'phase': 'idle', 'progress': 0, 'name': ''}
        return False
    # Step 2: replay frames at recorded timing
    teach_play_state = {'phase': 'playing', 'progress': 0, 'name': name}
    stop_flag = False
    moving = True
    total = len(frames)
    try:
        t0 = time.time()
        for i, frame in enumerate(frames):
            teach_play_state['progress'] = round(i / total * 100)
            if stop_flag:
                break
            target_t = frame['t']
            while time.time() - t0 < target_t:
                if stop_flag:
                    break
                time.sleep(0.005)
            q = {int(mid): val for mid, val in frame['q'].items()}
            grav = compute_gravity_torques(q)
            for mid in motors:
                if motor_enabled.get(mid, False):
                    try:
                        with _can_lock:
                            comm.MIT(mid, angle=q.get(mid, hold_pos.get(mid, 0.0)), velocity=0,
                                     torque=grav.get(mid, 0.0),
                                     kp=motor_kp[mid], kd=motor_kd[mid])
                    except Exception:
                        pass
    finally:
        if not stop_flag and frames:
            last_q = {int(mid): val for mid, val in frames[-1]['q'].items()}
            for mid in motors:
                hold_pos[mid] = last_q.get(mid, hold_pos[mid])
        moving = False
        teach_play_state['phase'] = 'idle'
        teach_play_state['progress'] = 100
    return True

_GRIPPER_RAMP_STEP = 0.5 / 25.0  # 0.5 rad/s at 25 Hz ≈ 1.1°/step → 90° in ~3s

def _hold_loop():
    """Background thread: continuously send MIT hold commands when not moving."""
    global _gripper_ramp_target, _gripper_torque_limit, _gripper_grasped
    while True:
        time.sleep(0.04)  # 25 Hz — faster hold reduces pre-motion sag window
        if moving or not hold_pos or comm is None:
            continue

        # ── Gripper ramp (runs inside hold_loop — no separate thread) ──
        if not teach_mode and _gripper_ramp_target is not None and 7 in hold_pos and motor_enabled.get(7, False):
            cur = hold_pos[7]
            target = _gripper_ramp_target
            with _gripper_lock:
                if _gripper_torque_limit is not None and abs(motors[7].state_tau) >= _gripper_torque_limit:
                    print(f"Gripper grasped: τ={motors[7].state_tau:.2f}Nm @ {math.degrees(cur):.1f}°")
                    _gripper_ramp_target = None
                    _gripper_torque_limit = None
                    _gripper_grasped = True
                    _gripper_done.set()
                else:
                    diff = target - cur
                    if abs(diff) <= _GRIPPER_RAMP_STEP:
                        hold_pos[7] = target
                        _gripper_ramp_target = None
                        _gripper_torque_limit = None
                        _gripper_done.set()
                    else:
                        hold_pos[7] = cur + math.copysign(_GRIPPER_RAMP_STEP, diff)

        if teach_mode:
            q_actual = {mid: m.state_q for mid, m in motors.items()}
            grav = compute_gravity_torques(q_actual)
        else:
            grav = compute_gravity_torques(dict(hold_pos))
        for mid, q in list(hold_pos.items()):
            if moving:
                break
            if motor_enabled.get(mid, False):
                try:
                    with _can_lock:
                        if teach_mode:
                            # Use actual position as target so kp doesn't spring back to hold_pos
                            actual_q = motors[mid].state_q
                            teach_kp = 2.0 if mid == 2 else 0
                            if mid == 2:
                                t_grav = grav[mid] * 1.22
                            elif mid == 3:
                                t_grav = grav[mid] * 1.52
                            else:
                                t_grav = grav.get(mid, 0.0)
                            comm.MIT(mid, angle=actual_q, velocity=0, torque=t_grav,
                                     kp=teach_kp, kd=0.3)
                        else:
                            comm.MIT(mid, angle=q, velocity=0, torque=grav.get(mid, 0.0),
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
                    comm.MIT(mid, angle=start[mid], velocity=0, torque=grav_start.get(mid, 0.0),
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
            print(f"J{mid} start out of limits ({start[mid]*180/math.pi:.1f}°), clamping")
            start[mid] = targets_rad.get(mid, max(lo_rad, min(hi_rad, start[mid])))

    duration = 3.0
    dt = 0.01
    t_start = time.time()

    try:
        while not stop_flag:
            t = time.time() - t_start
            if t >= duration:
                t = duration
            alpha   = 0.5 * (1 - math.cos(math.pi * t / duration))
            v_scale = 0.5 * (math.pi / duration) * math.sin(math.pi * t / duration)
            q_now = {mid: m.state_q for mid, m in motors.items()}
            grav = compute_gravity_torques(q_now)
            for mid in motors:
                delta = targets_rad.get(mid, start[mid]) - start[mid]
                q_des = start[mid] + alpha * delta
                v_des = v_scale * delta
                try:
                    with _can_lock:
                        comm.MIT(mid, angle=q_des, velocity=v_des, torque=grav.get(mid, 0.0),
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
            q_settle = {mid: targets_rad.get(mid, start[mid]) for mid in motors}
            grav_final = compute_gravity_torques(q_settle)
            while time.time() < settle_end:
                for mid in motors:
                    if motor_enabled.get(mid, False):
                        try:
                            with _can_lock:
                                comm.MIT(mid, angle=q_settle[mid], velocity=0,
                                         torque=grav_final.get(mid, 0.0),
                                         kp=motor_kp[mid], kd=motor_kd[mid])
                        except Exception:
                            pass
                time.sleep(0.02)  # 50 Hz settle rate
        # Hold at target positions so motor continues converging after trajectory ends
        for mid in motors:
            if mid in targets_rad:
                hold_pos[mid] = targets_rad[mid]
        moving = False
    return True

def gripper_command(target_rad, torque_limit=None):
    """Queue a gripper move — executed by hold_loop (no separate thread, no CAN conflicts)."""
    global _gripper_ramp_target, _gripper_torque_limit
    if 7 not in motors or not motor_enabled.get(7, False):
        return
    lo, hi = JOINT_LIMITS_DEG.get(7, (0, 90))
    target_rad = max(lo * math.pi / 180, min(hi * math.pi / 180, target_rad))
    _gripper_done.clear()
    with _gripper_lock:
        _gripper_torque_limit = torque_limit
        _gripper_ramp_target = target_rad

GRIPPER_GRASP_TORQUE = 1.2   # Nm — stop closing when contact detected
GRIPPER_OPEN_RAD = 90.0 * math.pi / 180

# ── Pick & Place Sequence ─────────────────────────────────────────

SEQUENCE_STEPS = [
    "移动到抓取接近位",
    "移动到抓取位",
    "夹爪夹取",
    "后退至抓取接近位",
    "移动到放置接近位",
    "移动到放置位",
    "夹爪张开",
    "后退至放置接近位",
]

def _get_approach_dir(orientation):
    """Return end-effector Z-axis in world frame for the given orientation."""
    if orientation == 'vertical':
        R = euler_to_rotation_matrix(np.pi, 0.0, 0.0)
    else:  # horizontal
        R = euler_to_rotation_matrix(-np.pi/2, 0.0, -np.pi/2)
    return R[:, 2]

def _sequence_ik(x_mm, y_mm, z_mm, orientation, q0=None):
    """Solve IK, return (joint_dict, joint_angles) or (None, None) on failure."""
    if q0 is None:
        q0 = np.array([motors[i+1].state_q if i+1 in motors else 0.0 for i in range(6)])
    T = _ik_target_transform(x_mm, y_mm, z_mm, orientation)
    result = inverse_kinematics(
        T, initial_guess=q0, use_urdf=True,
        joint_limits_lower=_IK_Q_LO, joint_limits_upper=_IK_Q_HI,
        max_iterations=500,
    )
    if result.success:
        return {i+1: float(result.joint_angles[i]) for i in range(6)}, result.joint_angles
    return None, None

def _sequence_preview(pick_x, pick_y, pick_z, pick_orient,
                      place_x, place_y, place_z, place_orient,
                      approach_dist_mm):
    """Pre-validate all 4 IK positions. Returns dict with results."""
    pick_dir  = _get_approach_dir(pick_orient)
    place_dir = _get_approach_dir(place_orient)
    d = approach_dist_mm

    positions = [
        ('pick_approach', pick_x  - pick_dir[0]*d,  pick_y  - pick_dir[1]*d,  pick_z  - pick_dir[2]*d,  pick_orient),
        ('pick',          pick_x,                    pick_y,                    pick_z,                    pick_orient),
        ('place_approach',place_x - place_dir[0]*d, place_y - place_dir[1]*d, place_z - place_dir[2]*d, place_orient),
        ('place',         place_x,                   place_y,                   place_z,                   place_orient),
    ]

    results = {}
    q0 = np.array([motors[i+1].state_q if i+1 in motors else 0.0 for i in range(6)])
    for key, x, y, z, orient in positions:
        T = _ik_target_transform(x, y, z, orient)
        r = inverse_kinematics(T, initial_guess=q0, use_urdf=True,
                               joint_limits_lower=_IK_Q_LO, joint_limits_upper=_IK_Q_HI,
                               max_iterations=500)
        if r.success:
            T_fk = forward_kinematics(r.joint_angles, use_urdf=True)
            results[key] = {
                'ok': True,
                'angles_deg': {str(i+1): round(float(np.degrees(r.joint_angles[i])), 1) for i in range(6)},
                'xyz': {'x': round(float(T_fk[0,3])*1000,1), 'y': round(float(T_fk[1,3])*1000,1), 'z': round(float(T_fk[2,3])*1000,1)},
                'pos_error_mm': round(r.position_error * 1000, 1),
            }
            q0 = r.joint_angles
        else:
            results[key] = {'ok': False, 'error': f'IK无解，位置误差 {r.position_error*1000:.0f}mm'}

    all_ok = all(v['ok'] for v in results.values())
    return {'ok': all_ok, 'positions': results}

def _run_sequence(pick_x, pick_y, pick_z, pick_orient,
                  place_x, place_y, place_z, place_orient,
                  approach_dist_mm):
    global _sequence_state, _sequence_stop_flag, _gripper_grasped

    def _step(n, name):
        _sequence_state.update({'step': n, 'step_name': name, 'error': None})
        print(f"[SEQ] Step {n}/8: {name}")

    def _fail(msg):
        _sequence_state.update({'active': False, 'error': msg})
        print(f"[SEQ] FAILED: {msg}")

    def _should_stop():
        return _sequence_stop_flag or stop_flag

    try:
        _sequence_state = {'active': True, 'step': 0, 'total_steps': 8,
                           'step_name': '准备中，求解IK…', 'error': None}

        # Compute approach xyz (offset along gripper direction, in mm)
        pick_dir  = _get_approach_dir(pick_orient)
        place_dir = _get_approach_dir(place_orient)
        d = approach_dist_mm

        q0 = np.array([motors[i+1].state_q if i+1 in motors else 0.0 for i in range(6)])

        pick_ap_j,  q0 = _sequence_ik(pick_x  - pick_dir[0]*d,  pick_y  - pick_dir[1]*d,  pick_z  - pick_dir[2]*d,  pick_orient,  q0)
        if pick_ap_j  is None: return _fail("IK无解：抓取接近位")

        pick_j,     q0 = _sequence_ik(pick_x,  pick_y,  pick_z,  pick_orient,  q0)
        if pick_j     is None: return _fail("IK无解：抓取位")

        place_ap_j, q0 = _sequence_ik(place_x - place_dir[0]*d, place_y - place_dir[1]*d, place_z - place_dir[2]*d, place_orient, q0)
        if place_ap_j is None: return _fail("IK无解：放置接近位")

        place_j,    q0 = _sequence_ik(place_x, place_y, place_z, place_orient, q0)
        if place_j    is None: return _fail("IK无解：放置位")

        # Step 1: move to pick approach
        if _should_stop(): return _fail("已中止")
        _step(1, SEQUENCE_STEPS[0])
        execute_move(dict(pick_ap_j))
        if stop_flag: return _fail("急停中止")

        # Step 2: move to pick position
        if _should_stop(): return _fail("已中止")
        _step(2, SEQUENCE_STEPS[1])
        execute_move(dict(pick_j))
        if stop_flag: return _fail("急停中止")

        # Step 3: grasp
        if _should_stop(): return _fail("已中止")
        _step(3, SEQUENCE_STEPS[2])
        _gripper_grasped = False
        gripper_command(0.0, torque_limit=GRIPPER_GRASP_TORQUE)
        _gripper_done.wait(timeout=8.0)
        if not _gripper_grasped:
            return _fail("⚠ 夹取失败：未检测到夹持力矩，请检查目标位置")

        # Step 4: retreat
        if _should_stop(): return _fail("已中止")
        _step(4, SEQUENCE_STEPS[3])
        execute_move(dict(pick_ap_j))
        if stop_flag: return _fail("急停中止")

        # Step 5: move to place approach
        if _should_stop(): return _fail("已中止")
        _step(5, SEQUENCE_STEPS[4])
        execute_move(dict(place_ap_j))
        if stop_flag: return _fail("急停中止")

        # Step 6: move to place position
        if _should_stop(): return _fail("已中止")
        _step(6, SEQUENCE_STEPS[5])
        execute_move(dict(place_j))
        if stop_flag: return _fail("急停中止")

        # Step 7: open gripper
        if _should_stop(): return _fail("已中止")
        _step(7, SEQUENCE_STEPS[6])
        gripper_command(GRIPPER_OPEN_RAD)
        _gripper_done.wait(timeout=8.0)

        # Step 8: retreat from place
        if _should_stop(): return _fail("已中止")
        _step(8, SEQUENCE_STEPS[7])
        execute_move(dict(place_ap_j))

        _sequence_state.update({'active': False, 'step': 8, 'step_name': '✓ 完成', 'error': None})
        print("[SEQ] Completed successfully.")

    except Exception as e:
        _fail(f"序列异常: {e}")

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
        global _ik_preview_result, _sequence_stop_flag
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
                hold_pos[mid] = motors[mid].state_q  # hold at current position, not old target
            self._json_response({'ok': True})

        elif path == '/api/enable_motor':
            data = json.loads(body)
            mid = int(data['id'])
            for _ in range(3):
                comm.Enable(mid)
                time.sleep(0.01)
            motor_enabled[mid] = True
            hold_pos[mid] = motors[mid].state_q  # hold at current position, not old target
            self._json_response({'ok': True, 'id': mid, 'enabled': True})

        elif path == '/api/disable_motor':
            data = json.loads(body)
            mid = int(data['id'])
            comm.Disable(mid)
            motor_enabled[mid] = False
            self._json_response({'ok': True, 'id': mid, 'enabled': False})

        elif path == '/api/status':
            result = {}
            for mid, m in motors.items():
                result[str(mid)] = {'q': m.state_q, 'tau': m.state_tau}
            self._json_response(result)

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

        elif path == '/api/teach/start':
            ok = teach_start()
            self._json_response({'ok': ok})

        elif path == '/api/teach/stop':
            entry = teach_stop()
            if entry:
                self._json_response({'ok': True, 'name': entry['name'],
                                     'duration': entry['duration'], 'count': entry['count']})
            else:
                self._json_response({'ok': False, 'error': 'Recording too short (<0.5s)'})

        elif path == '/api/teach/list':
            summary = [{'name': t['name'], 'duration': t['duration'], 'count': t['count']}
                       for t in teachings]
            self._json_response(summary)

        elif path == '/api/teach/play':
            data = json.loads(body)
            idx = int(data['index'])
            if 0 <= idx < len(teachings):
                t = teachings[idx]
                thread = threading.Thread(target=play_teaching,
                                          args=(t['frames'], t['name']), daemon=True)
                thread.start()
                self._json_response({'ok': True})
            else:
                self._json_response({'ok': False, 'error': 'Invalid index'})

        elif path == '/api/teach/play_status':
            self._json_response(dict(teach_play_state))

        elif path == '/api/teach/rename':
            data = json.loads(body)
            idx = int(data['index'])
            if 0 <= idx < len(teachings):
                teachings[idx]['name'] = data['name']
                _save_teachings()
                self._json_response({'ok': True})
            else:
                self._json_response({'ok': False, 'error': 'Invalid index'})

        elif path == '/api/teach/delete':
            data = json.loads(body)
            idx = int(data['index'])
            if 0 <= idx < len(teachings):
                teachings.pop(idx)
                _save_teachings()
                self._json_response({'ok': True})
            else:
                self._json_response({'ok': False, 'error': 'Invalid index'})

        elif path == '/api/ik_preview':
            data = json.loads(body)
            x_mm = float(data['x'])
            y_mm = float(data['y'])
            z_mm = float(data['z'])
            orientation = data.get('orientation', 'horizontal')
            q0 = np.array([motors[i+1].state_q if i+1 in motors else 0.0
                           for i in range(6)])
            T_target = _ik_target_transform(x_mm, y_mm, z_mm, orientation)
            result = inverse_kinematics(
                T_target, initial_guess=q0, use_urdf=True,
                joint_limits_lower=_IK_Q_LO, joint_limits_upper=_IK_Q_HI,
                max_iterations=500,
            )
            if result.success:
                T_actual = forward_kinematics(result.joint_angles, use_urdf=True)
                angles_deg = {str(i+1): round(float(np.degrees(result.joint_angles[i])), 1)
                              for i in range(6)}
                targets = {i+1: float(result.joint_angles[i]) for i in range(6)}
                with _ik_lock:
                    _ik_preview_result = targets
                self._json_response({
                    'ok': True,
                    'joint_angles_deg': angles_deg,
                    'actual_xyz': {
                        'x': round(float(T_actual[0, 3]) * 1000, 1),
                        'y': round(float(T_actual[1, 3]) * 1000, 1),
                        'z': round(float(T_actual[2, 3]) * 1000, 1),
                    },
                    'pos_error_mm': round(result.position_error * 1000, 1),
                })
            else:
                with _ik_lock:
                    _ik_preview_result = None
                self._json_response({
                    'ok': False,
                    'error': f'IK无解，位置误差 {result.position_error*1000:.0f}mm，目标可能超出工作空间',
                })

        elif path == '/api/ik_execute':
            with _ik_lock:
                targets = dict(_ik_preview_result) if _ik_preview_result else None
            if targets is None:
                self._json_response({'ok': False, 'error': '请先执行预览'})
            else:
                thread = threading.Thread(target=execute_move, args=(targets,), daemon=True)
                thread.start()
                self._json_response({'ok': True})

        elif path == '/api/gripper_grasp':
            # Close gripper until torque limit hit (object detected) or fully closed
            gripper_command(0.0, torque_limit=GRIPPER_GRASP_TORQUE)
            self._json_response({'ok': True})

        elif path == '/api/gripper_open':
            gripper_command(GRIPPER_OPEN_RAD)
            self._json_response({'ok': True})

        elif path == '/api/gripper_move':
            data = json.loads(body)
            angle_deg = float(data['angle_deg'])
            lo, hi = JOINT_LIMITS_DEG.get(7, (0, 90))
            angle_deg = max(lo, min(hi, angle_deg))
            angle_rad = angle_deg * math.pi / 180
            gripper_command(angle_rad)
            self._json_response({'ok': True, 'angle_deg': round(angle_deg, 1)})

        elif path == '/api/ee_pose':
            q_arr = np.zeros(6)
            for i in range(6):
                mid = i + 1
                if mid in motors:
                    q_arr[i] = motors[mid].state_q
            T = forward_kinematics(q_arr, use_urdf=True)
            rpy = rotation_matrix_to_euler(T[:3, :3])
            self._json_response({
                'x':     round(float(T[0, 3]) * 1000, 1),
                'y':     round(float(T[1, 3]) * 1000, 1),
                'z':     round(float(T[2, 3]) * 1000, 1),
                'roll':  round(float(np.degrees(rpy[0])), 1),
                'pitch': round(float(np.degrees(rpy[1])), 1),
                'yaw':   round(float(np.degrees(rpy[2])), 1),
            })

        elif path == '/api/sequence/preview':
            data = json.loads(body)
            result = _sequence_preview(
                float(data['pick_x']),  float(data['pick_y']),  float(data['pick_z']),
                data.get('pick_orient', 'horizontal'),
                float(data['place_x']), float(data['place_y']), float(data['place_z']),
                data.get('place_orient', 'horizontal'),
                float(data.get('approach_dist', 100)),
            )
            self._json_response(result)

        elif path == '/api/sequence/run':
            data = json.loads(body)
            if _sequence_state.get('active', False):
                self._json_response({'ok': False, 'error': '序列正在运行中'})
            else:
                _sequence_stop_flag = False
                thread = threading.Thread(
                    target=_run_sequence,
                    kwargs={
                        'pick_x':   float(data['pick_x']),  'pick_y':   float(data['pick_y']),
                        'pick_z':   float(data['pick_z']),  'pick_orient': data.get('pick_orient','horizontal'),
                        'place_x':  float(data['place_x']), 'place_y':  float(data['place_y']),
                        'place_z':  float(data['place_z']), 'place_orient': data.get('place_orient','horizontal'),
                        'approach_dist_mm': float(data.get('approach_dist', 100)),
                    },
                    daemon=True)
                thread.start()
                self._json_response({'ok': True})

        elif path == '/api/sequence/stop':
            _sequence_stop_flag = True
            emergency_stop()
            self._json_response({'ok': True})

        elif path == '/api/sequence/status':
            self._json_response(dict(_sequence_state))

        else:
            self.send_error(404)

# ── Main ─────────────────────────────────────────────────────────

def main():
    os.makedirs(os.path.join(BASE_DIR, 'web'), exist_ok=True)

    _load_teachings()
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
