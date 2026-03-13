# pkg_robotarm_py/Publisher.py
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import random
import time
import signal
import sys
from std_msgs.msg import Float64MultiArray

def signal_handler(sig, frame):
    print('\n收到中断信号，正在关闭...')
    sys.exit(0)

class Angel_Publisher(Node):
    def __init__(self,name):
        super().__init__(name)

        self.pub = self.create_publisher(Float64MultiArray, 'target_angle', 10)
        self.timer = self.create_timer(0.001,self.pub_msg)

    def pub_msg(self):
        msg = Float64MultiArray()

        msg.data = [random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)]
        # self.get_logger().info(f'发送的数据: {msg.data}')

        self.pub.publish(msg)
    
def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill命令
    rclpy.init()
    pub_demo = Angel_Publisher("publisher_node")
    rclpy.spin(pub_demo)
    pub_demo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

