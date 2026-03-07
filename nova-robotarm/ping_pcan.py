"""Ping RobStride motors via PCAN-USB adapter (PCANBasic.dll)."""
import ctypes
import time
import sys
import os

# --- PCANBasic constants ---
PCAN_USBBUS1 = 0x51
PCAN_BAUD_1M = 0x0014
PCAN_ERROR_OK = 0x00000
PCAN_ERROR_QRCVEMPTY = 0x00020
PCAN_MESSAGE_EXTENDED = 0x02

class TPCANMsg(ctypes.Structure):
    _fields_ = [
        ("ID", ctypes.c_uint32),
        ("MSGTYPE", ctypes.c_ubyte),
        ("LEN", ctypes.c_ubyte),
        ("DATA", ctypes.c_ubyte * 8),
    ]

class TPCANTimestamp(ctypes.Structure):
    _fields_ = [
        ("millis", ctypes.c_uint32),
        ("millis_overflow", ctypes.c_ushort),
        ("micros", ctypes.c_ushort),
    ]

def load_pcan():
    candidates = [
        r"C:\Windows\System32\PCANBasic.dll",
        r"C:\Program Files\PEAK-System\PEAK PCAN-Basic API\x64\PCANBasic.dll",
        "PCANBasic.dll",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ctypes.windll.LoadLibrary(path)
    return ctypes.windll.LoadLibrary("PCANBasic.dll")

pcan = load_pcan()

SERVER_ID = 0xFD

# Motor type limits: [max_angle_rad, max_vel, max_torque]
LIMIT_PARAM = {
    '00': [12.57, 33, 14],    # 4310
    '03': [12.57, 20, 60],    # 4340
    '04': [12.57, 15, 120],   # 4310_48
}

# Motor config: motor_id -> type (matching client.py)
MOTOR_TYPES = {
    1: '04', 2: '04',  # J1, J2: 4310_48
    3: '03',            # J3: 4340
    4: '00', 5: '00', 6: '00',  # J4-J6: 4310
}

def parse_response(msg):
    """Parse CAN feedback frame from RobStride motor.

    Response CAN ID (29-bit):
      [31:24] comm_type
      [23:22] mode (2 bits)
      [21:16] fault (6 bits)
      [15:8]  motor_id
      [7:0]   host_id (0xFD)
    """
    can_id = msg.ID
    comm_type = (can_id >> 24) & 0xFF
    motor_id = (can_id >> 8) & 0xFF
    host_id = can_id & 0xFF
    data = bytes(msg.DATA[:msg.LEN])

    if comm_type == 2 and len(data) >= 8:
        mode_bits = (can_id >> 22) & 0x03
        fault_bits = (can_id >> 16) & 0x3F

        angle_raw = (data[0] << 8) | data[1]
        vel_raw = (data[2] << 8) | data[3]
        torque_raw = (data[4] << 8) | data[5]
        temp_raw = (data[6] << 8) | data[7]

        mt = MOTOR_TYPES.get(motor_id, '00')
        lim = LIMIT_PARAM.get(mt, [12.57, 33, 14])

        angle = (angle_raw / 65535 - 0.5) * 2 * lim[0]
        velocity = (vel_raw / 65535 - 0.5) * 2 * lim[1]
        torque = (torque_raw / 65535 - 0.5) * 2 * lim[2]
        temperature = temp_raw / 10.0

        mode_names = {0: "reset", 1: "cali", 2: "run", 3: "error"}
        mode = mode_names.get(mode_bits, f"?{mode_bits}")

        faults = []
        if fault_bits & 0x20: faults.append("uncal")
        if fault_bits & 0x10: faults.append("hall")
        if fault_bits & 0x08: faults.append("mag")
        if fault_bits & 0x04: faults.append("overT")
        if fault_bits & 0x02: faults.append("overI")
        if fault_bits & 0x01: faults.append("underV")
        fault_str = ",".join(faults) if faults else "ok"

        print(f"  [Motor {motor_id}] type={mt}  pos={angle:+8.3f}rad  vel={velocity:+7.3f}r/s  "
              f"torque={torque:+6.2f}Nm  temp={temperature:5.1f}C  mode={mode}  fault={fault_str}")
        return motor_id
    else:
        print(f"  [?] comm={comm_type} ID=0x{can_id:08X} data={data.hex()}")
        return None

def main():
    print("=== PCAN-USB Motor Ping ===\n")

    result = pcan.CAN_Initialize(PCAN_USBBUS1, PCAN_BAUD_1M, 0, 0, 0)
    if result != PCAN_ERROR_OK:
        print(f"CAN_Initialize failed: 0x{result:05X}")
        sys.exit(1)
    print("PCAN-USB initialized (1 Mbit/s)\n")

    # Flush
    while True:
        msg = TPCANMsg()
        ts = TPCANTimestamp()
        if pcan.CAN_Read(PCAN_USBBUS1, ctypes.byref(msg), ctypes.byref(ts)) == PCAN_ERROR_QRCVEMPTY:
            break

    # Send Enable to motors 1~16
    max_id = 16
    print(f"Sending Enable to motor IDs 1~{max_id} ...\n")
    for motor_id in range(1, max_id + 1):
        msg = TPCANMsg()
        msg.ID = (0x03 << 24) | (SERVER_ID << 8) | motor_id
        msg.MSGTYPE = PCAN_MESSAGE_EXTENDED
        msg.LEN = 8
        for i in range(8):
            msg.DATA[i] = 0
        pcan.CAN_Write(PCAN_USBBUS1, ctypes.byref(msg))
        time.sleep(0.02)

    # Read responses
    print("Responses:\n")
    found = set()
    deadline = time.time() + 1.5
    while time.time() < deadline:
        msg = TPCANMsg()
        ts = TPCANTimestamp()
        result = pcan.CAN_Read(PCAN_USBBUS1, ctypes.byref(msg), ctypes.byref(ts))
        if result == PCAN_ERROR_OK:
            mid = parse_response(msg)
            if mid:
                found.add(mid)
        elif result == PCAN_ERROR_QRCVEMPTY:
            time.sleep(0.001)
        else:
            break

    pcan.CAN_Uninitialize(PCAN_USBBUS1)
    print(f"\n=== Found {len(found)} motor(s): {sorted(found)} ===")

if __name__ == "__main__":
    main()
