from rclpy.node import Node
import time
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Twist, PoseStamped, Vector3
from rosgraph_msgs.msg import Clock
from custom_interfaces.msg import EnvsProperties
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from tb3_multi_env_spawner import utils


class ROSInterface(Node):
    def __init__(self, namespace: str):
        super().__init__(namespace=namespace, node_name='ros_interface')
        self.namespace = namespace
        print(f"{self.namespace}, PORCOIDDIOOOOOOOOOOOOOOOOoo")


        self.map_msg = None
        self.scan_msg = None
        self.tf_msg = None
        self.tf_map2odom = None
        self.tf_odom2foot = None
        self.gazebo_clock_msg = None
        self.env_properties = None

        clock_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.map_sub = self.create_subscription(OccupancyGrid, f'/{self.namespace}/map', self._map_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, f'/{self.namespace}/scan', self._scan_callback, 10)
        self.tf_sub = self.create_subscription(TFMessage, f'/{self.namespace}/tf', self._tf_callback, 100)
        self.gazebo_clock_sub = self.create_subscription(Clock, '/clock', self._gazebo_clock_callback, clock_qos_profile)
        self.envs_properties_sub = self.create_subscription(EnvsProperties, '/envs_properties', self._envs_properties_callback, 10)

        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.namespace}/cmd_vel', 10)
        self.goal_pose_stamped_pub = self.create_publisher(PoseStamped, f'/{self.namespace}/goal_pose', 10)


    ## MAP DATA ##

    def _map_callback(self, msg):
        self.map_msg = OccupancyGrid()
        self.map_msg = msg

    def get_map_area(self):
        return self.map_msg.info.width * self.map_msg.info.height
    
    def get_map_data(self):
        return self.map_msg.data
    

    ## SCAN DATA ##

    def _scan_callback(self, msg):
        self.scan_msg = LaserScan()
        self.scan_msg = msg

    def get_scan_ranges(self):
        return self.scan_msg.ranges

    ## POSE DATA ##

    def _tf_callback(self, msg):
        self.tf_msg = TFMessage()
        for transform in msg.transforms:
            if transform.header.frame_id == 'map' and transform.child_frame_id == 'odom':
                self.tf_map2odom = transform.transform
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_footprint':
                self.tf_odom2foot = transform.transform
    

    def get_robot_pose(self):
        if self.tf_odom2foot is not None and self.tf_map2odom is not None:
            from scipy.spatial.transform import Rotation as R

            # Extract position and quaternion
            p_o2f = np.array([self.tf_odom2foot.translation.x,
                            self.tf_odom2foot.translation.y])

            q_o2f = np.array([self.tf_odom2foot.rotation.x,
                            self.tf_odom2foot.rotation.y,
                            self.tf_odom2foot.rotation.z,
                            self.tf_odom2foot.rotation.w])

            p_m2o = np.array([self.tf_map2odom.translation.x,
                            self.tf_map2odom.translation.y])

            q_m2o = np.array([self.tf_map2odom.rotation.x,
                            self.tf_map2odom.rotation.y,
                            self.tf_map2odom.rotation.z,
                            self.tf_map2odom.rotation.w])

            # Convert quaternions to rotation matrices
            R_m2o = R.from_quat(q_m2o).as_matrix()
            R_o2f = R.from_quat(q_o2f).as_matrix()

            # Compute robot position
            robot_position = p_m2o + R_m2o[:2, :2] @ p_o2f  # Only use 2D components

            # Compute yaw relative to the map frame
            yaw_m2o = R.from_quat(q_m2o).as_euler('xyz', degrees=False)[2]  # Yaw of odom in map
            yaw_o2f = R.from_quat(q_o2f).as_euler('xyz', degrees=False)[2]  # Yaw of base in odom
            robot_angle = yaw_m2o + yaw_o2f  # Combine yaw angles

            # Ensure angle is in [-pi, pi]
            robot_angle = (robot_angle + np.pi) % (2 * np.pi) - np.pi  

            # Alternative: Use sin/cos encoding
            robot_angle_sin = np.sin(robot_angle)
            robot_angle_cos = np.cos(robot_angle)


            self.robot_pose = [robot_position[0], robot_position[1], robot_angle]

            # Debug print for first environment
            if self.namespace == 'env_0':
                print(f"[{self.namespace}] robot_pose: {self.robot_pose}")
                print("cos, sin: ", robot_angle_cos, robot_angle_sin)
            return self.robot_pose
        
        else:
            raise ValueError('tf_map2odom or tf_odom2foot is None')
        


    ## GAZEBO TIME DATA ##

    def _gazebo_clock_callback(self, msg):
        self.gazebo_clock_msg = Clock()
        self.gazebo_clock_msg = msg

    def get_gazebo_time(self):
        return self.gazebo_clock_msg.clock.sec + self.gazebo_clock_msg.clock.nanosec * 1e-9

    ## CMD_VEL ##

    def publish_action(self, action):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = action[0].astype(float)
        cmd_vel_msg.angular.z = action[1].astype(float)
        self.cmd_vel_pub.publish(cmd_vel_msg)


    ## ENVS DATA ##

    def _envs_properties_callback(self, msg):
        envs_properties = EnvsProperties()
        envs_properties = msg
        for properties in envs_properties.data:
            if properties.name == self.namespace:
                self.env_properties = properties
                break


    ## POSE DATA ##
    def create_pose_stamped(self, frame_id, pose):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = np.sin(pose[2]/2)  # sin(yaw/2)
        pose_msg.pose.orientation.w = np.cos(pose[2]/2)  # cos(yaw/2)
        return pose_msg

    ## GOAL POSE ##

    def set_random_goal_pose(self):
        center = (self.env_properties.center[0], self.env_properties.center[1])
        self.goal_pose = utils.get_random_pose(self.env_properties.file_path, center)
        self.publish_goal_pose()
        return self.goal_pose

    def publish_goal_pose(self):
        goal_pose_stamped = self.create_pose_stamped(frame_id='map', pose=self.goal_pose)
        self.goal_pose_stamped_pub.publish(goal_pose_stamped)

    ## RVIZ MARKERS ##
    # def publish_goal_pose(self):
    #     marker_array = MarkerArray()
    #     sphere_scale = Vector3(x=0.1, y=0.1, z=0.1)
    #     arrow_scale = Vector3(x=0.2, y=0.05, z=0.05)
    #     color_green = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    #     goal_pose_stamped = self.create_pose_stamped(frame_id='odom', pose=[0.0,0.0,0.0])
    #     position_sphere = self.create_marker('goal_pose', 0, Marker.SPHERE, Marker.ADD,
    #                                    goal_pose_stamped, sphere_scale, color_green)
    #     orientation_arrow = self.create_marker('goal_pose', 1, Marker.ARROW, Marker.ADD,
    #                                    goal_pose_stamped, arrow_scale, color_green)
    #     marker_array.markers.append(position_sphere)
    #     marker_array.markers.append(orientation_arrow)
    #     self.marker_array_pub.publish(marker_array)


    # def create_marker(self, ns, id, type, action, pose, scale, color):
    #     marker = Marker()
    #     marker.header = pose.header
    #     marker.ns = ns
    #     marker.id = id
    #     marker.type = type
    #     marker.action = action
    #     marker.pose = pose.pose
    #     marker.scale = scale
    #     marker.color = color

    #     return marker
    
