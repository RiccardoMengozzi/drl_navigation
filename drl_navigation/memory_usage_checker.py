import rclpy
import time
import psutil

from rclpy.node import Node

from std_msgs.msg import Bool
from std_srvs.srv import Trigger



MEMORY_USAGE_THRESHOLD = 75.0 # gzserver memory leak workaround
LOG_PERIOD = 10

class MemoryUsageChecker(Node):
    def __init__(self):
        super().__init__('memory_usage_checker')
        self.is_resetting = False
        self.count = 0

        self.reset_gazebo_pub = self.create_publisher(Bool, '/reset_gazebo_topic', 10)
        self.reset_gazebo_cli = self.create_client(Trigger, '/reset_gazebo_service')

        self.create_timer(1.0, self.check_memory_usage)

    def check_memory_usage(self):   
        self.count += 1
        self.reset_gazebo_pub.publish(Bool(data=self.is_resetting)) 
        if self.count % LOG_PERIOD == 0: 
            self.count = 0
            self.get_logger().info(f"Memory usage: {psutil.virtual_memory().percent}%")

        # Check if the meory percentage usage is above the threshold (gzserver memory leak workaround)
        if psutil.virtual_memory().percent > MEMORY_USAGE_THRESHOLD and not self.is_resetting:
            self.get_logger().info(f"Memory usage is above {MEMORY_USAGE_THRESHOLD}%. Restarting gazebo ...")
            self.restart_gazebo()

    def restart_gazebo(self):
            self.is_resetting = True
            self.reset_gazebo_pub.publish(Bool(data=True))
            # wait for environments to be stopped in reset()
            time.sleep(5) # Could be improved by reading from a dedicated topic created by each environment....

            while not self.reset_gazebo_cli.wait_for_service(timeout_sec=5.0):
                self.get_logger().info(f'...waiting for reset_gazebo_service...')
                pass
            self.get_logger().info(f'...connected!')

            future = self.reset_gazebo_cli.call_async(Trigger.Request())
            future.add_done_callback(self.reset_gazebo_response_callback)

    def reset_gazebo_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info('Gazebo reset successful!')
                self.is_resetting = False

        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")




def main(args=None):
    try:
        rclpy.init(args=args)
        memory_usage_checker = MemoryUsageChecker()
        rclpy.spin(memory_usage_checker)

    except KeyboardInterrupt:
        memory_usage_checker.get_logger().info('Keyboard interrupt detected. Exiting...')

    finally:
        memory_usage_checker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()




