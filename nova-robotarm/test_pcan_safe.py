"""
Safe test: Enable → read state → Disable. NO movement commands.

This script:
  1. Opens PCAN-USB connection
  2. Enables each motor (puts it in 'run' mode, holds current position)
  3. Reads position/velocity/torque/temperature 10 times at 10Hz
  4. Disables all motors (back to 'reset' mode, motors go limp)
  5. Closes connection

Safe because: Enable without MIT/POS commands = motor holds position with no torque output.
"""
import sys
import os
import time

# Add grpc_stream to path so we can import robstride_pcan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'robot_arm', 'grpc_stream'))
from robstride_pcan import Motor, MotorControl

def main():
    # Motor config (same as client.py)
    motors = [
        Motor('04', 1, True),   # J1: 4310_48, inverse
        Motor('04', 2, True),   # J2: 4310_48, inverse
        Motor('03', 3, False),  # J3: 4340
        Motor('00', 4, False),  # J4: 4310
        Motor('00', 5, False),  # J5: 4310
        Motor('00', 6, True),   # J6: 4310, inverse
    ]

    comm = MotorControl()
    comm.opon_device()
    comm.clear_buffer()

    for m in motors:
        comm.addMotor(m)

    # Enable all motors (multiple times to ensure state transition)
    print("\n=== Enabling motors ===")
    for _ in range(3):
        for m in motors:
            comm.Enable(m.SlaveID)
        time.sleep(0.05)

    # Read state 10 times
    print("\n=== Reading motor states (10 iterations, 100ms interval) ===\n")
    print(f"{'#':>3}  {'J1':>8}  {'J2':>8}  {'J3':>8}  {'J4':>8}  {'J5':>8}  {'J6':>8}  (rad)")
    print("-" * 70)

    for i in range(10):
        # Send Enable to read state (Enable = read without movement)
        for m in motors:
            comm.Enable(m.SlaveID)

        positions = [f"{m.state_q:+8.4f}" for m in motors]
        print(f"{i+1:3d}  {'  '.join(positions)}")
        time.sleep(0.1)

    # Print detailed final state
    print("\n=== Final motor states ===\n")
    for m in motors:
        print(f"  J{m.SlaveID}: type={m.MotorType}  "
              f"pos={m.state_q:+8.4f} rad  "
              f"vel={m.state_dq:+7.3f} r/s  "
              f"torque={m.state_tau:+6.3f} Nm  "
              f"inverse={m.inverse}")

    # Disable all motors (go limp)
    print("\n=== Disabling motors ===")
    for m in motors:
        comm.Disable(m.SlaveID)
        time.sleep(0.02)

    comm.close_device()
    print("\nDone. All motors disabled safely.")

if __name__ == "__main__":
    main()
