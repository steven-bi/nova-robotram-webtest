# pkg_robotarm_py/Subscriber.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import signal
import sys
import time
from std_msgs.msg import Float64MultiArray
from . import robstride
from . import Inv_Dyn

def signal_handler(sig, frame):
    print('\n收到中断信号，正在关闭...')
    # 执行清理操作
    cleanup()
    sys.exit(0)

def cleanup():
    """资源清理函数"""
    print("正在关闭电机...")
    for i in range(6):
        try:
            Motor_control.Disable(i+1)
        except:
            print(f"禁用电机{i+1}失败")
    try:
        Motor_control.close_device()
    except:
        print("关闭设备失败")


class Angel_Subscriber(Node):
    def __init__(self,name):
        super().__init__(name)
        self.angle_data = None
        self.pub = self.create_publisher(Float64MultiArray, 'arm_angle', 10)
        self.sub = self.create_subscription(Float64MultiArray, 'target_angle', self.listener_callback,10)
        # time.sleep(4)
        self.timer = self.create_timer(0.01,self.Motor_control)           

    def listener_callback(self,msg):
        self.angle_data = msg.data
        self.get_logger().info(f'接收数据: {self.angle_data}')
    
    def Motor_control(self):
        # if self.angle_data == None :
        #     self.get_logger().info(f'还没收到数据')
        # else:
        #     self.get_logger().info(f'定时器接收的数据: {self.angle_data}')
        theta_deg = [Motor_1.smooth_q, Motor_2.smooth_q, Motor_3.smooth_q, Motor_4.smooth_q,Motor_5.smooth_q,Motor_6.smooth_q]      #六个关节对笛卡尔坐标系矫正后的角度
        theta_d_rad = [0,0,0,0,0,0]     #六个关节的角速度
        theta_dd_rad = [0,0,0,0,0,0]     #六个关节的角加速度
        torque=Inv_Dyn.Inv_Dyn(theta_deg, theta_d_rad, theta_dd_rad)      #正运动学与动力学计算出各个关节的力矩和主手末端的x,y,z坐标
        for idx in range(6):    #对主手各个关节下发控制指令
            Motor_control.MIT(idx+1,0,0,torque[idx],0,0)

        pub_msg = Float64MultiArray()
        pub_msg.data = [Motor_1.smooth_q, Motor_2.smooth_q, Motor_3.smooth_q, Motor_4.smooth_q,Motor_5.smooth_q,Motor_6.smooth_q]
        self.pub.publish(pub_msg)

Motor_Kp = [3,3,3,60,60,60]
Motor_Kd = [0.08,0.08,0.08,4,4,4]
Motor_1=robstride.Motor('04', 1, True)
Motor_2=robstride.Motor('04', 2, True)
Motor_3=robstride.Motor('03', 3, False)
Motor_4=robstride.Motor('00', 4, False)
Motor_5=robstride.Motor('00', 5, False)
Motor_6=robstride.Motor('00', 6, True)
Motor_list=[Motor_1,Motor_2,Motor_3,Motor_4,Motor_5,Motor_6]
Motor_control=robstride.MotorControl()

def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令

    rclpy.init()
    pub_demo = Angel_Subscriber("subscription_node")

    Motor_control.opon_device()
    Motor_control.clear_buffer()
    for j,motor in enumerate(Motor_list):
        Motor_control.addMotor(motor)
        Motor_control.Enable(j+1)
    time.sleep(0.5)

    rclpy.spin(pub_demo)

    for i in range(6):
        Motor_control.Disable(i+1)
    Motor_control.close_device()
    pub_demo.destroy_node()
    rclpy.shutdown()