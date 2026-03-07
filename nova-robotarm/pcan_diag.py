"""PCAN-USB diagnostic - check bus status and attempt communication."""
import ctypes
import time
import sys
import os

PCAN_USBBUS1 = 0x51
PCAN_ERROR_OK = 0x00000
PCAN_ERROR_QRCVEMPTY = 0x00020
PCAN_MESSAGE_EXTENDED = 0x02

# Baud rates to try
BAUD_RATES = {
    "1M":    0x0014,
    "500K":  0x001C,
    "250K":  0x011C,
    "125K":  0x031C,
}

# PCAN parameters
PCAN_ERROR_STATUS = 0x04  # Bus status parameter
PCAN_BUSOFF_AUTORESET = 0x07
PCAN_LISTEN_ONLY = 0x08

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

def get_status(channel):
    """Get PCAN channel error status."""
    status = ctypes.c_uint32()
    result = pcan.CAN_GetValue(channel, 0x04, ctypes.byref(status), 4)
    if result == PCAN_ERROR_OK:
        return status.value
    return result

def try_baud(baud_name, baud_value):
    """Try a specific baud rate: init, send, check for responses."""
    print(f"\n--- Trying {baud_name} (0x{baud_value:04X}) ---")

    result = pcan.CAN_Initialize(PCAN_USBBUS1, baud_value, 0, 0, 0)
    if result != PCAN_ERROR_OK:
        print(f"  Initialize failed: 0x{result:05X}")
        return False

    # Check initial bus status
    status = get_status(PCAN_USBBUS1)
    print(f"  Initial bus status: 0x{status:05X}")

    # Flush
    while True:
        msg = TPCANMsg()
        ts = TPCANTimestamp()
        if pcan.CAN_Read(PCAN_USBBUS1, ctypes.byref(msg), ctypes.byref(ts)) == PCAN_ERROR_QRCVEMPTY:
            break

    # Send Enable to motor 1 (most likely to exist)
    msg = TPCANMsg()
    msg.ID = (0x03 << 24) | (SERVER_ID << 8) | 1
    msg.MSGTYPE = PCAN_MESSAGE_EXTENDED
    msg.LEN = 8
    for i in range(8):
        msg.DATA[i] = 0

    result = pcan.CAN_Write(PCAN_USBBUS1, ctypes.byref(msg))
    print(f"  Write result: 0x{result:05X} ({'OK' if result == 0 else 'ERROR'})")

    time.sleep(0.1)

    # Check bus status after write
    status = get_status(PCAN_USBBUS1)
    print(f"  Bus status after write: 0x{status:05X}")

    # Read responses
    found = False
    deadline = time.time() + 0.5
    while time.time() < deadline:
        msg = TPCANMsg()
        ts = TPCANTimestamp()
        result = pcan.CAN_Read(PCAN_USBBUS1, ctypes.byref(msg), ctypes.byref(ts))
        if result == PCAN_ERROR_OK:
            can_id = msg.ID
            data = bytes(msg.DATA[:msg.LEN])
            print(f"  RECV: ID=0x{can_id:08X} LEN={msg.LEN} DATA={data.hex()} MSGTYPE=0x{msg.MSGTYPE:02X}")
            found = True
        elif result == PCAN_ERROR_QRCVEMPTY:
            time.sleep(0.005)
        else:
            print(f"  Read error: 0x{result:05X}")
            break

    if not found:
        print(f"  No response received")

    # Check final status
    status = get_status(PCAN_USBBUS1)
    print(f"  Final bus status: 0x{status:05X}")

    pcan.CAN_Uninitialize(PCAN_USBBUS1)
    return found

def main():
    print("=== PCAN-USB Diagnostic ===")
    print("Testing multiple baud rates...\n")

    for name, value in BAUD_RATES.items():
        success = try_baud(name, value)
        if success:
            print(f"\n*** SUCCESS at {name} ***")
            break
        time.sleep(0.2)
    else:
        print("\n\nNo response at any baud rate.")
        print("Check:")
        print("  1. CAN wires connected (CANH, CANL)")
        print("  2. Motors powered on")
        print("  3. Termination resistor (120 ohm) present")

if __name__ == "__main__":
    main()
