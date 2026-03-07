"""
Joint Control GUI — drag sliders, click Execute, robot moves smoothly.

Features:
  - 6 sliders for J1-J6 (range ±180°)
  - "Read Current" button: sync sliders to actual motor positions
  - "Execute" button: smoothly interpolate from current to target (2s)
  - "STOP" button: immediately disable all motors
  - Real-time position display

Safety:
  - Linear interpolation over 2 seconds (no sudden jumps)
  - Moderate MIT gains (kp=10~30, kd=0.2~2)
  - Motors auto-disable on window close
"""
import sys
import os
import math
import time
import threading
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robot_arm', 'grpc_stream'))
from robstride_pcan import Motor, MotorControl

# ── Config ───────────────────────────────────────────────────────

MOTOR_CONFIG = [
    # (id, type, inverse, kp, kd, label)
    (1, '04', True,  15, 0.3, 'J1 base'),
    (2, '04', True,  15, 0.3, 'J2 shoulder'),
    (3, '03', False, 15, 0.3, 'J3 elbow'),
    (4, '00', False, 30, 1.0, 'J4 wrist1'),
    (5, '00', False, 30, 1.0, 'J5 wrist2'),
    (6, '00', True,  30, 1.0, 'J6 wrist3'),
]

MOVE_DURATION = 2.0  # seconds to interpolate
CONTROL_HZ = 100

# ── App ──────────────────────────────────────────────────────────

class JointControlApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Robot Arm Joint Control")
        self.root.geometry("520x580")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.comm = None
        self.motors = {}
        self.motor_kp = {}
        self.motor_kd = {}
        self.moving = False
        self.stop_flag = False

        self._init_hardware()
        self._build_ui()
        self._read_current()

    # ── Hardware ─────────────────────────────────────────────────

    def _init_hardware(self):
        self.comm = MotorControl()
        self.comm.opon_device()
        self.comm.clear_buffer()

        for mid, mtype, inv, kp, kd, label in MOTOR_CONFIG:
            m = Motor(mtype, mid, inv)
            self.comm.addMotor(m)
            self.motors[mid] = m
            self.motor_kp[mid] = kp
            self.motor_kd[mid] = kd

        # Enable all
        for _ in range(3):
            for mid in self.motors:
                self.comm.Enable(mid)
            time.sleep(0.03)

    def _shutdown_hardware(self):
        if self.comm:
            for mid in self.motors:
                try:
                    self.comm.Disable(mid)
                except:
                    pass
            try:
                self.comm.close_device()
            except:
                pass

    # ── UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title
        title = tk.Label(self.root, text="Robot Arm Joint Control",
                         font=("Arial", 16, "bold"))
        title.pack(pady=(10, 5))

        # Joint sliders frame
        frame = tk.Frame(self.root)
        frame.pack(padx=15, pady=5, fill='x')

        self.sliders = {}
        self.slider_labels = {}
        self.current_labels = {}

        for i, (mid, mtype, inv, kp, kd, label) in enumerate(MOTOR_CONFIG):
            row = tk.Frame(frame)
            row.pack(fill='x', pady=3)

            # Joint name
            tk.Label(row, text=f"J{mid}", font=("Arial", 12, "bold"),
                     width=3).pack(side='left')

            # Current position label
            cur_lbl = tk.Label(row, text="  0.0°", font=("Consolas", 10),
                               fg="gray", width=7, anchor='e')
            cur_lbl.pack(side='left', padx=(0, 5))
            self.current_labels[mid] = cur_lbl

            # Arrow
            tk.Label(row, text="→", font=("Arial", 10)).pack(side='left')

            # Slider
            var = tk.DoubleVar(value=0)
            slider = tk.Scale(row, from_=-180, to=180, orient='horizontal',
                              variable=var, resolution=0.5, length=250,
                              showvalue=False, command=lambda v, m=mid: self._on_slider(m))
            slider.pack(side='left', padx=5)
            self.sliders[mid] = var

            # Target label
            tgt_lbl = tk.Label(row, text="  0.0°", font=("Consolas", 10, "bold"),
                               width=7, anchor='e')
            tgt_lbl.pack(side='left')
            self.slider_labels[mid] = tgt_lbl

        # Buttons frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=15)

        self.read_btn = tk.Button(btn_frame, text="Read Current",
                                  font=("Arial", 11), width=14,
                                  command=self._read_current)
        self.read_btn.pack(side='left', padx=5)

        self.exec_btn = tk.Button(btn_frame, text="Execute",
                                  font=("Arial", 11, "bold"), width=14,
                                  bg="#4CAF50", fg="white",
                                  command=self._execute)
        self.exec_btn.pack(side='left', padx=5)

        self.stop_btn = tk.Button(btn_frame, text="STOP",
                                  font=("Arial", 11, "bold"), width=14,
                                  bg="#f44336", fg="white",
                                  command=self._emergency_stop)
        self.stop_btn.pack(side='left', padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status = tk.Label(self.root, textvariable=self.status_var,
                          font=("Arial", 10), fg="gray", anchor='w')
        status.pack(padx=15, pady=(5, 0), fill='x')

        # Progress bar
        self.progress = ttk.Progressbar(self.root, length=480, mode='determinate')
        self.progress.pack(padx=15, pady=(5, 10))

    def _on_slider(self, mid):
        val = self.sliders[mid].get()
        self.slider_labels[mid].config(text=f"{val:+6.1f}°")

    def _update_current_labels(self):
        for mid, m in self.motors.items():
            deg = m.state_q * 180 / math.pi
            self.current_labels[mid].config(text=f"{deg:+6.1f}°")

    # ── Actions ──────────────────────────────────────────────────

    def _read_current(self):
        """Read current motor positions and sync sliders."""
        for _ in range(3):
            for mid in self.motors:
                self.comm.Enable(mid)
            time.sleep(0.02)

        for mid, m in self.motors.items():
            deg = m.state_q * 180 / math.pi
            self.sliders[mid].set(round(deg, 1))
            self.slider_labels[mid].config(text=f"{deg:+6.1f}°")

        self._update_current_labels()
        self.status_var.set("Synced sliders to current positions")

    def _execute(self):
        """Smoothly move to slider targets."""
        if self.moving:
            return

        # Read current positions
        for mid in self.motors:
            self.comm.Enable(mid)
        time.sleep(0.02)

        # Collect start and target
        start = {}
        target = {}
        for mid, m in self.motors.items():
            start[mid] = m.state_q
            target[mid] = self.sliders[mid].get() * math.pi / 180.0

        # Check max delta
        max_delta_deg = max(abs(target[mid] - start[mid]) * 180 / math.pi
                           for mid in self.motors)
        if max_delta_deg < 0.5:
            self.status_var.set("Already at target (< 0.5° change)")
            return

        self.moving = True
        self.stop_flag = False
        self.exec_btn.config(state='disabled')
        self.read_btn.config(state='disabled')
        self.status_var.set(f"Moving... (max delta: {max_delta_deg:.1f}°)")

        # Run control loop in thread
        thread = threading.Thread(target=self._control_loop,
                                  args=(start, target), daemon=True)
        thread.start()

    def _control_loop(self, start, target):
        """Interpolate from start to target over MOVE_DURATION seconds."""
        dt = 1.0 / CONTROL_HZ
        t_start = time.time()

        try:
            while not self.stop_flag:
                t = time.time() - t_start
                if t >= MOVE_DURATION:
                    t = MOVE_DURATION

                # Linear interpolation with smooth easing (cosine)
                alpha = t / MOVE_DURATION
                # Smooth step: 0→1 with ease-in-out
                alpha = 0.5 * (1 - math.cos(math.pi * alpha))

                for mid in self.motors:
                    q_des = start[mid] + alpha * (target[mid] - start[mid])
                    dq_des = 0  # let MIT handle velocity
                    self.comm.MIT(mid, angle=q_des, velocity=dq_des, torque=0,
                                  kp=self.motor_kp[mid], kd=self.motor_kd[mid])

                # Update progress on main thread
                progress_pct = min(100, t / MOVE_DURATION * 100)
                self.root.after(0, self._update_progress, progress_pct)

                if t >= MOVE_DURATION:
                    break

                time.sleep(dt)

        except Exception as e:
            self.root.after(0, self.status_var.set, f"Error: {e}")

        # Done
        self.root.after(0, self._move_complete)

    def _update_progress(self, pct):
        self.progress['value'] = pct
        for mid in self.motors:
            self.comm.Enable(mid)
        self._update_current_labels()

    def _move_complete(self):
        self.moving = False
        self.exec_btn.config(state='normal')
        self.read_btn.config(state='normal')
        self.progress['value'] = 100 if not self.stop_flag else 0

        if self.stop_flag:
            self.status_var.set("STOPPED by user")
        else:
            self.status_var.set("Move complete!")
            self._update_current_labels()

    def _emergency_stop(self):
        """Immediately stop all motors."""
        self.stop_flag = True
        for mid in self.motors:
            try:
                self.comm.Disable(mid)
            except:
                pass
        self.status_var.set("EMERGENCY STOP — all motors disabled")
        self.moving = False
        self.exec_btn.config(state='normal')
        self.read_btn.config(state='normal')
        self.progress['value'] = 0

    def _on_close(self):
        self.stop_flag = True
        time.sleep(0.1)
        self._shutdown_hardware()
        self.root.destroy()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = JointControlApp()
    app.run()
