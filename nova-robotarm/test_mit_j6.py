"""
Safe MIT test on J6 only.

Phase 1 (3s): Lock at current position with soft kp=3, kd=0.1
              Motor gently holds position. You can try pushing it by hand.
Phase 2: Auto-disable, motor goes limp.

Ctrl+C to abort at any time.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robot_arm', 'grpc_stream'))
from robstride_pcan import Motor, MotorControl

def main():
    # Only J6
    motor6 = Motor('00', 6, True)

    comm = MotorControl()
    comm.opon_device()
    comm.clear_buffer()
    comm.addMotor(motor6)

    # Enable a few times to ensure state transition
    print("Enabling J6 ...")
    for _ in range(4):
        comm.Enable(6)
        time.sleep(0.05)

    q0 = motor6.state_q
    print(f"J6 current position: {q0:+.4f} rad ({q0*180/3.14159:+.1f} deg)")
    print()
    print(f"=== Phase 1: MIT hold at current position (3 seconds) ===")
    print(f"    target={q0:+.4f} rad, kp=3, kd=0.1, torque=0")
    print(f"    Motor should gently resist if you push it.")
    print()

    dt = 0.01  # 10ms control loop = 100Hz
    duration = 3.0
    t_start = time.time()

    try:
        while True:
            t = time.time() - t_start
            if t > duration:
                break

            # MIT: hold at q0 with soft gains
            comm.MIT(6, angle=q0, velocity=0, torque=0, kp=3, kd=0.1)

            if int(t * 10) % 5 == 0:  # print every ~0.5s
                print(f"  t={t:.1f}s  pos={motor6.state_q:+.4f} rad  "
                      f"vel={motor6.state_dq:+.3f} r/s  "
                      f"torque={motor6.state_tau:+.3f} Nm")

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n  [Ctrl+C] Aborting!")

    # Always disable
    print("\n=== Disabling J6 ===")
    comm.Disable(6)
    comm.close_device()
    print("Done. J6 is limp now.")

if __name__ == "__main__":
    main()
