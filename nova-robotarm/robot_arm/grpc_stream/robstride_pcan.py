"""
RobStride motor driver — PCANBasic.dll backend.

Drop-in replacement for robstride.py (which uses ControlCAN.dll / ZLGCAN).
Only MotorControl class is changed; Motor class is identical.

API mapping:
  ZLGCAN                    → PCANBasic
  VCI_OpenDevice+InitCAN    → CAN_Initialize
  VCI_Transmit              → CAN_Write
  VCI_Receive               → CAN_Read (loop)
  VCI_ClearBuffer           → read-until-empty
  VCI_CloseDevice           → CAN_Uninitialize
"""

from enum import IntEnum
from ctypes import *
import ctypes
import time
import sys
import os
import struct
import math
from collections import deque

# ── PCANBasic constants ──────────────────────────────────────────
PCAN_USBBUS1        = 0x51
PCAN_BAUD_1M        = 0x0014
PCAN_ERROR_OK        = 0x00000
PCAN_ERROR_QRCVEMPTY = 0x00020
PCAN_MESSAGE_EXTENDED = 0x02

class TPCANMsg(Structure):
    _fields_ = [
        ("ID",      c_uint32),
        ("MSGTYPE", c_ubyte),
        ("LEN",     c_ubyte),
        ("DATA",    c_ubyte * 8),
    ]

class TPCANTimestamp(Structure):
    _fields_ = [
        ("millis",          c_uint32),
        ("millis_overflow", c_ushort),
        ("micros",          c_ushort),
    ]

def _load_pcan_dll():
    candidates = [
        r"C:\Windows\System32\PCANBasic.dll",
        r"C:\Program Files\PEAK-System\PEAK PCAN-Basic API\x64\PCANBasic.dll",
        "PCANBasic.dll",
    ]
    for path in candidates:
        if os.path.exists(path):
            return windll.LoadLibrary(path)
    return windll.LoadLibrary("PCANBasic.dll")

# ── Motor (unchanged from original) ─────────────────────────────

class Control_Type(IntEnum):
    MIT = 0
    POS = 1
    POS_VEL = 5
    VEL = 2
    Torque_Pos = 3

class Motor:
    def __init__(self, MotorType, SlaveID, inverse):
        self.Pd = 0.0
        self.Vd = 0.0
        self.state_q = 0.0
        self.state_dq = 0.0
        self.state_tau = 0.0
        self.alpha = 0.8
        self.smooth_q = 0.0
        self.smooth_dq = 0.0
        self.smooth_tau = 0.0
        self.SlaveID = SlaveID
        self.MotorType = MotorType
        self.NowControlMode = Control_Type.MIT
        self.inverse = inverse

    def recv_data(self, q, dq, tau):
        self.state_q = q
        self.state_dq = dq
        self.state_tau = tau
        self.smooth_q = self.alpha * q + (1 - self.alpha) * self.smooth_q
        self.smooth_dq = self.alpha * dq + (1 - self.alpha) * self.smooth_dq
        self.smooth_tau = self.alpha * tau + (1 - self.alpha) * self.smooth_tau

    def getPosition(self):
        return self.state_q

    def getVelocity(self):
        return self.state_dq

    def getTorque(self):
        return self.state_tau

# ── MotorControl (PCAN backend) ─────────────────────────────────

class MotorControl:

    def __init__(self, can=1) -> None:
        self.control_dict = {
            'GetID': 0x0, 'MIT': 0x1, 'Enable': 0x3, 'Stop': 0x4,
            'SetZero': 0x6, 'SetID': 0x7,
            'GetInfo': 0x11, 'SetInfo': 0x12, 'GetError': 0x15,
            'SetBaud': 0x16, 'Save': 0x18, 'Server_ID': 0xfd,
        }
        #                   4310              4310_48           4340
        self.Limit_Param = {
            '00': [12.57, 33, 14],
            '01': [12.57, 44, 17],
            '02': [12.57, 44, 17],
            '03': [12.57, 20, 60],
            '04': [12.57, 15, 120],
            '05': [12.57, 50, 5.5],
            '06': [12.57, 50, 36],
        }
        self.motors_map = {}
        self._pcan = _load_pcan_dll()
        self._channel = PCAN_USBBUS1
        print('init robot communication (PCAN backend)')

    # ── Device open / close ──────────────────────────────────────

    def opon_device(self):
        # Retry loop: PCAN driver needs time to release after previous process exits
        for attempt in range(10):
            self._pcan.CAN_Uninitialize(self._channel)
            time.sleep(0.5)
            ret = self._pcan.CAN_Initialize(self._channel, PCAN_BAUD_1M, 0, 0, 0)
            if ret == PCAN_ERROR_OK:
                break
            print(f'  CAN_Initialize attempt {attempt+1}: 0x{ret:05X}, retrying...')
            time.sleep(1.0)
        assert ret == PCAN_ERROR_OK, f'CAN_Initialize failed after retries: 0x{ret:05X}'
        print('open communication device (PCAN-USB, 1 Mbit/s)')

    def close_device(self):
        self._pcan.CAN_Uninitialize(self._channel)
        print('close communication device (PCAN-USB)')

    def clear_buffer(self):
        count = 0
        while True:
            msg = TPCANMsg()
            ts = TPCANTimestamp()
            ret = self._pcan.CAN_Read(self._channel, byref(msg), byref(ts))
            if ret == PCAN_ERROR_QRCVEMPTY:
                break
            count += 1
        print(f'clear buffer (flushed {count} frames)')

    # ── Low-level send / receive ─────────────────────────────────

    def send_message(self, ID, DATA):
        msg = TPCANMsg()
        msg.ID = ID
        msg.MSGTYPE = PCAN_MESSAGE_EXTENDED
        msg.LEN = 8
        for i in range(8):
            msg.DATA[i] = DATA[i] if i < len(DATA) else 0
        ret = self._pcan.CAN_Write(self._channel, byref(msg))
        assert ret == PCAN_ERROR_OK, f'CAN_Write failed: 0x{ret:05X}'

    def recivice_message_once(self):
        """Read available CAN frames (non-blocking, up to 50 frames)."""
        messages = []
        for _ in range(50):
            msg = TPCANMsg()
            ts = TPCANTimestamp()
            ret = self._pcan.CAN_Read(self._channel, byref(msg), byref(ts))
            if ret == PCAN_ERROR_QRCVEMPTY:
                break
            if ret == PCAN_ERROR_OK:
                messages.append({
                    'ID': hex(msg.ID),
                    'DataLen': hex(msg.LEN),
                    'Data': list(msg.DATA[:msg.LEN]),
                })
        return messages

    def recivice_message_wait(self, timeout_ms=100):
        """Read with retry until at least one frame or timeout."""
        deadline = time.time() + timeout_ms / 1000.0
        while time.time() < deadline:
            msgs = self.recivice_message_once()
            if msgs:
                return msgs
            time.sleep(0.001)
        return []

    # ── Motor management ─────────────────────────────────────────

    def addMotor(self, motor):
        self.motors_map[motor.SlaveID] = motor
        return True

    # ── Commands (same logic as original) ────────────────────────

    def Enable(self, id):
        ubyte_array = c_ubyte * 8
        ID_code = (self.control_dict['Enable'] << 24) | (self.control_dict['Server_ID'] << 8) | id
        self.send_message(ID_code, ubyte_array())
        message = self.recivice_message_wait()
        for msg in message:
            self.decoder(msg)
        return None

    def Disable(self, id, Clear=False):
        ubyte_array = c_ubyte * 8
        data = ubyte_array()
        ID_code = (self.control_dict['Stop'] << 24) | (self.control_dict['Server_ID'] << 8) | id
        if Clear:
            data[0] = 1
        self.send_message(ID_code, data)
        message = self.recivice_message_wait()
        for msg in message:
            self.decoder(msg)
        return None

    def Set_Zero(self, id):
        ubyte_array = c_ubyte * 8
        data = ubyte_array()
        ID_code = (self.control_dict['SetZero'] << 24) | (self.control_dict['Server_ID'] << 8) | id
        data[0] = 1
        self.send_message(ID_code, data)
        message = self.recivice_message_wait()
        for msg in message:
            self.decoder(msg)

    def Set_Mode(self, id, mode=0):
        ubyte_array = c_ubyte * 8
        data = ubyte_array()
        ID_code = (self.control_dict['SetInfo'] << 24) | (self.control_dict['Server_ID'] << 8) | id
        data[:2] = self.int_to_bytes_array(0x7005, inverse=True)
        data[4] = mode
        self.motors_map[id].Control_Type = mode
        self.send_message(ID_code, data)
        message = self.recivice_message_wait()
        time.sleep(0.1)
        for msg in message:
            print(self.decoder(msg))

    def MIT(self, id, angle, velocity, torque, kp, kd):
        if self.motors_map[id].inverse:
            angle = -angle
            velocity = -velocity
            torque = -torque
        MotorType = self.motors_map[id].MotorType
        P_max, V_max, T_max = self.Limit_Param[MotorType]
        angle_int = int(((self.clip(angle, -P_max, P_max) + P_max) / (2 * P_max)) * 65535)
        velocity_int = int(((self.clip(velocity, -V_max, V_max) + V_max) / (2 * V_max)) * 65535)
        torque_int = int(((self.clip(torque, -T_max, T_max) + T_max) / (2 * T_max)) * 65535)
        kp_int = int((self.clip(kp, 0, 500) / 500.0) * 65535)
        kd_int = int((self.clip(kd, 0, 5) / 5.0) * 65535)
        ubyte_array = c_ubyte * 8
        data = ubyte_array()
        ID_code = (self.control_dict['MIT'] << 24) | (torque_int << 8) | id
        data[:2] = self.int_to_bytes_array(angle_int)
        data[2:4] = self.int_to_bytes_array(velocity_int)
        data[4:6] = self.int_to_bytes_array(kp_int)
        data[6:8] = self.int_to_bytes_array(kd_int)
        self.send_message(ID_code, data)
        message = self.recivice_message_once()
        for msg in message:
            self.decoder(msg)

    def POS_velocity(self, id, pos=0.0, vel=0.0):
        ubyte_array = c_ubyte * 8
        data = ubyte_array()
        # Set velocity limit first
        data[:2] = self.int_to_bytes_array(0x7017, inverse=True)
        data[2], data[3] = 0x00, 0x00
        for i, b in enumerate(struct.pack("<f", vel)):
            data[4 + i] = b
        ID_code = (self.control_dict['SetInfo'] << 24) | (self.control_dict['Server_ID'] << 8) | id
        self.send_message(ID_code, data)
        # Set position target
        data[:2] = self.int_to_bytes_array(0x7016, inverse=True)
        data[2], data[3] = 0x00, 0x00
        for i, b in enumerate(struct.pack("<f", pos)):
            data[4 + i] = b
        self.send_message(ID_code, data)
        message = self.recivice_message_wait()
        for msg in message:
            self.decoder(msg)

    # ── Decoder ──────────────────────────────────────────────────

    def decoder(self, message):
        hex_number = int(message['ID'], 16)
        Data = message['Data']
        control_ID = (hex_number >> 24) & 0xFF
        if control_ID == 2:
            motor_can_id = (hex_number >> 8) & 0xFF
            host_can_id = hex_number & 0xFF
            fault_info = (hex_number >> 16) & 0x3F
            angle = self.bytes_array_to_int(Data[:2])
            velocity = self.bytes_array_to_int(Data[2:4])
            torque = self.bytes_array_to_int(Data[4:6])
            temperature = self.bytes_array_to_int(Data[6:8])
            MotorType = self.motors_map[motor_can_id].MotorType
            motor_info = {
                'angle': (angle / 65535 - 0.5) * 2 * self.Limit_Param[MotorType][0],
                'velocity': (velocity / 65535 - 0.5) * 2 * self.Limit_Param[MotorType][1],
                'torque': (torque / 65535 - 0.5) * 2 * self.Limit_Param[MotorType][2],
                'temperature': temperature / 10,
            }
            if self.motors_map[motor_can_id].inverse:
                motor_info['angle'] = -motor_info['angle']
                motor_info['velocity'] = -motor_info['velocity']
                motor_info['torque'] = -motor_info['torque']
            self.motors_map[motor_can_id].recv_data(
                motor_info['angle'], motor_info['velocity'], motor_info['torque'])
            return motor_info
        else:
            print(f'response: control_ID={control_ID}, data={Data}')

    # ── Helpers ──────────────────────────────────────────────────

    def clip(self, val, min_val, max_val):
        return max(min_val, min(max_val, val))

    def int_to_bytes_array(self, decimal_number, inverse=False):
        if inverse:
            return [decimal_number & 0xFF, (decimal_number >> 8) & 0xFF]
        else:
            return [(decimal_number >> 8) & 0xFF, decimal_number & 0xFF]

    def bytes_array_to_int(self, byte_array, inverse=False):
        if inverse:
            return byte_array[0] | (byte_array[1] << 8)
        else:
            return (byte_array[0] << 8) | byte_array[1]
