
import time
import numpy as np
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
from gazebo_msgs.srv import SetEntityState

from rclpy.qos import QoSProfile, QoSReliabilityPolicy


class RosInterface(Node):
    def __init__(self) -> None:
        super().__init__("ros_interface")

        self.get_logger().info("ROS Interface node started")

        self.cb_group = ReentrantCallbackGroup()

        self.scan_msg = None
        self.odom_msg = None
        self.gazebo_clock_msg = None

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self._scan_callback, 10, callback_group=self.cb_group)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_callback, 10, callback_group=self.cb_group)

        clock_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.gazebo_clock_sub = self.create_subscription(Clock, '/clock', self._gazebo_clock_callback, clock_qos_profile, callback_group=self.cb_group)

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.goal_point_pub = self.create_publisher(MarkerArray, "/goal_point", 10)

        self.reset_world_cli = self.create_client(Empty, "/reset_world")
        self.set_entity_state_cli = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.pause_simulation_cli = self.create_client(Empty, "/pause_physics")
        self.unpause_simulation_cli = self.create_client(Empty, "/unpause_physics")


    def _quaternion_to_euler(self, x: float, y: float, z: float, w: float) -> float:
        '''Converts quaternion to euler'''
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = np.arctan2(t3, t4)

        return roll_x, pitch_y, yaw_z

    ### GAZEBO CLOCK ###
    def _gazebo_clock_callback(self, msg: Clock) -> None:
        self.gazebo_clock_msg = Clock()
        self.gazebo_clock_msg = msg

    def get_gazebo_time(self) -> float:
        '''Returns the time from the gazebo clock'''
        return self.gazebo_clock_msg.clock.sec + self.gazebo_clock_msg.clock.nanosec * 1e-9


    ### SCAN ###

    def _scan_callback(self, msg: LaserScan) -> None:
        self.scan_msg = LaserScan()
        self.scan_msg = msg

    def get_scan_ranges(self) -> np.array:
        '''Returns the ranges of the scan message'''
        return np.array(self.scan_msg.ranges)
    
    def get_scan_min_max_range(self) -> np.array:
        '''Returns the minimum and maximum range of the scan message'''
        return np.array([self.scan_msg.range_min, self.scan_msg.range_max])
    
    def get_scan_angle_increment(self) -> float:
        '''Returns the angle increment of the scan message'''
        return self.scan_msg.angle_increment
    
    def get_scan_angle_min(self) -> float:
        '''Returns the minimum angle of the scan message'''
        return self.scan_msg.angle_min
    
    def get_scan_angle_max(self) -> float:
        '''Returns the maximum angle of the scan message'''
        return self.scan_msg.angle_max
    
    def get_scan_time_increment(self) -> float:
        '''Returns the time increment of the scan message'''
        return self.scan_msg.time_increment
    
    ### ODOM ###

    def _odom_callback(self, msg: Odometry) -> None:
        self.odom_msg = Odometry()
        self.odom_msg = msg

    def get_odom_position(self) -> np.array:
        '''Returns the position of the odometry message'''
        return np.array([self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y])
    
    def get_odom_orientation(self) -> float:
        '''Returns the orientation of the odometry message'''
        _, _, yaw = self._quaternion_to_euler(self.odom_msg.pose.pose.orientation.x,
                                              self.odom_msg.pose.pose.orientation.y,
                                              self.odom_msg.pose.pose.orientation.z,
                                              self.odom_msg.pose.pose.orientation.w)
        return yaw
    
    def get_odom_linear_velocity(self) -> np.array:
        '''Returns the linear velocity of the odometry message'''
        return np.array([self.odom_msg.twist.twist.linear.x, 
                         self.odom_msg.twist.twist.linear.y, 
                         self.odom_msg.twist.twist.linear.z])
    
    def get_odom_angular_velocity(self) -> np.array:
        '''Returns the angular velocity of the odometry message'''
        return np.array([self.odom_msg.twist.twist.angular.x, 
                         self.odom_msg.twist.twist.angular.y,
                         self.odom_msg.twist.twist.angular.z])

    ## CMD VEL ##

    def publish_cmd_vel(self, linear_x: float, angular_z: float) -> None:
        '''Publishes the cmd_vel message'''
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_vel_pub.publish(msg)

    ### GOAL POINT ###

    def publish_goal_point(self, goal_position: np.array) -> None:
        '''Publishes the goal point'''
        x = goal_position[0]
        y = goal_position[1]
        msg = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        msg.markers.append(marker)
        self.goal_point_pub.publish(msg)

    ### RESET WORLD ###
    def reset_world(self) -> None:
        '''Resets the world'''
        # self.get_logger().info("Waiting for'/reset_world service...")
        while not self.reset_world_cli.wait_for_service(timeout_sec=1.0):
            pass
        # self.get_logger().info("...Connected!")
        req = Empty.Request()
        future = self.reset_world_cli.call_async(req)
        while not future.done():
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
        
        if future.result() is not None:
            pass
            # self.get_logger().info("World reset successfully")
        else:
            self.get_logger().error("Failed to reset world")

    ### SET ENTITY STATE ###
    def set_entity_state(self, entity_name: str, x: float, y: float, yaw: float) -> None:
        '''Sets the entity state'''
        # self.get_logger().info("Waiting for '/gazebo/set_entity_state' service...")
        while not self.set_entity_state_cli.wait_for_service(timeout_sec=1.0):
            pass
        # self.get_logger().info("...Connected!")
        req = SetEntityState.Request()
        req.state.name = entity_name
        req.state.pose.position.x = x
        req.state.pose.position.y = y
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = np.sin(yaw / 2)
        req.state.pose.orientation.w = np.cos(yaw / 2)
        future = self.set_entity_state_cli.call_async(req)
        while not future.done():
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
        if future.result() is not None:
            pass
            # self.get_logger().info(f"Entity state set successfully: {entity_name}")
        else:
            self.get_logger().error(f"Failed to set entity state: {entity_name}")


    ### PAUSE SIMULATION ###

    def pause_simulation(self) -> None:
        '''Pauses the simulation'''
        # self.get_logger().info("Waiting for '/pause_physics' service...")
        while not self.pause_simulation_cli.wait_for_service(timeout_sec=1.0):
            pass
        # self.get_logger().info("...Connected!")
        req = Empty.Request()
        future = self.pause_simulation_cli.call_async(req)
        while not future.done():
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
        
        if future.result() is not None:
            pass
            # self.get_logger().info("Simulation paused successfully")
        else:
            self.get_logger().error("Failed to pause simulation")


    ### UNPAUSE SIMULATION ###

    def unpause_simulation(self) -> None:
        '''Unpauses the simulation'''
        # self.get_logger().info("Waiting for '/unpause_physics' service...")
        while not self.unpause_simulation_cli.wait_for_service(timeout_sec=1.0):
            pass
        # self.get_logger().info("...Connected!")
        req = Empty.Request()
        future = self.unpause_simulation_cli.call_async(req)
        while not future.done():
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
        
        if future.result() is not None:
            pass
            # self.get_logger().info("Simulation unpaused successfully")
        else:
            self.get_logger().error("Failed to unpause simulation")


