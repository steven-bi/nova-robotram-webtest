"""
J6 sine wave test: ±15° (30° total) around current position.

Slow sine: 0.3Hz, amplitude=15°=0.2618rad, kp=3, kd=0.1
Duration: 10 seconds (~3 full cycles)
Auto-disable after completion.

Ctrl+C to abort at any time.
"""
import sys
import os
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robot_arm', 'grpc_stream'))
from robstride_pcan import Motor, MotorControl

def main():
    motor6 = Motor('00', 6, True)

    comm = MotorControl()
    comm.opon_device()
    comm.clear_buffer()
    comm.addMotor(motor6)

    print("Enabling J6 ...")
    for _ in range(4):
        comm.Enable(6)
        time.sleep(0.05)

    q0 = motor6.state_q
    print(f"J6 current position: {q0:+.4f} rad ({q0*180/math.pi:+.1f} deg)")

    # Sine parameters
    amplitude = 15.0 * math.pi / 180.0  # 15° = 0.2618 rad
    freq = 0.3       # Hz (slow)
    omega = 2 * math.pi * freq
    kp = 3
    kd = 0.1
    duration = 10.0   # ~3 full cycles
    dt = 0.01         # 100Hz control loop

    print()
    print(f"=== Sine wave: ±15° at 0.3Hz for {duration:.0f}s ===")
    print(f"    center={q0:+.4f} rad, amplitude={amplitude:.4f} rad ({amplitude*180/math.pi:.1f} deg)")
    print(f"    kp={kp}, kd={kd}")
    print()

    t_start = time.time()
    last_print = -1.0

    try:
        while True:
            t = time.time() - t_start
            if t > duration:
                break

            q_des = q0 + amplitude * math.sin(omega * t)
            dq_des = amplitude * omega * math.cos(omega * t)

            comm.MIT(6, angle=q_des, velocity=dq_des, torque=0, kp=kp, kd=kd)

            # Print every 0.5s
            if t - last_print >= 0.5:
                err = motor6.state_q - q_des
                print(f"  t={t:4.1f}s  target={q_des:+.4f}  actual={motor6.state_q:+.4f}  "
                      f"err={err:+.4f} rad ({err*180/math.pi:+.1f} deg)  "
                      f"torque={motor6.state_tau:+.3f} Nm")
                last_print = t

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n  [Ctrl+C] Aborting!")

    print("\n=== Disabling J6 ===")
    comm.Disable(6)
    comm.close_device()
    print("Done. J6 is limp now.")

if __name__ == "__main__":
    main()
