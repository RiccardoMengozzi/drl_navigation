from rclpy.node import Node
import time
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
from rosgraph_msgs.msg import Clock
from custom_interfaces.msg import EnvsProperties
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA



class ROSInterface(Node):
    def __init__(self, namespace: str):
        super().__init__(namespace=namespace, node_name='ros_interface')
        self.namespace = namespace


        self.map_msg = None
        self.scan_msg = None
        self.tf_msg = None
        self.tf_map2odom = None
        self.tf_odom2foot = None
        self.robot_pose = None
        self.gazebo_clock_msg = None
        self.env_properties = None

        ### DEBUGGING ###
        self.chosen_goals_list = PoseArray()
        self.chosen_goals_list_pub = self.create_publisher(PoseArray, f'/{self.namespace}/goals_pose_list', 10)
        #################

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

            # Extract position and quaternion
            self.p_o2f = np.array([self.tf_odom2foot.translation.x,
                            self.tf_odom2foot.translation.y])
                            

            q_o2f = np.array([self.tf_odom2foot.rotation.x,
                            self.tf_odom2foot.rotation.y,
                            self.tf_odom2foot.rotation.z,
                            self.tf_odom2foot.rotation.w])

            self.p_m2o = np.array([self.tf_map2odom.translation.x,
                            self.tf_map2odom.translation.y])
            
            q_m2o = np.array([self.tf_map2odom.rotation.x,
                            self.tf_map2odom.rotation.y,
                            self.tf_map2odom.rotation.z,
                            self.tf_map2odom.rotation.w])

            # Convert quaternions to rotation matrices
            self.R_m2o = R.from_quat(q_m2o).as_matrix()
            self.R_o2f = R.from_quat(q_o2f).as_matrix()

            # Compute robot position
            robot_position = self.p_m2o + self.R_m2o[:2, :2] @ self.p_o2f  # Only use 2D components

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
            print(f"[{self.namespace}] robot_pose: {self.robot_pose}")

            # # Debug print for first environment
            # if self.namespace == 'env_0':
            #     print(f"[{self.namespace}] robot_pose: {self.robot_pose}")
            #     print("cos, sin: ", robot_angle_cos, robot_angle_sin)
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
        # actually the pose should be respect the 'odom' frame, that is in the center of the map,
        # however rviz will show the marker only if the frame_id is 'map', so the position is compensated.
        pose_msg.header.frame_id = frame_id
        pose_msg.pose.position.x = pose[0]
        pose_msg.pose.position.y = pose[1]
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = np.sin(pose[2]/2)  # sin(yaw/2)
        pose_msg.pose.orientation.w = np.cos(pose[2]/2)  # cos(yaw/2)
        return pose_msg

    ## GOAL POSE ##

    def set_random_goal_pose(self, max_distance):
        x_start = self.env_properties.x_start
        y_start = self.env_properties.y_start
        resolution = self.env_properties.resolution
        x_center = self.env_properties.center[0]
        y_center = self.env_properties.center[1]
        

        # Restore 2d grid from 1d flatten array
        grid = np.array(self.env_properties.grid).reshape(self.env_properties.grid_size)
        # Find the indices of the cells where the value is 1 (points inside the environment free of obstacles)
        indices = np.argwhere(grid == 1)
        
        goal_distance = 100000
        while goal_distance > max_distance:
            # Select a random index and remove it after (to speed up the search process)
            random_index = random.choice(indices)
            indices = np.delete(indices, np.where((indices == random_index).all(axis=1)), axis=0)
            # goal_pose wrt to "odom" frame_id
            goal_position_odom = [x_start + random_index[0] * resolution + x_center, 
                                  y_start + random_index[1] * resolution + y_center] 
            # goal_pose wrt to "map"
            goal_position_map = self.R_m2o[:2,:2] @ goal_position_odom[:2] + self.p_m2o 
            goal_distance = np.linalg.norm(goal_position_map - self.robot_pose[:2])

        random_yaw = np.random.uniform(0, 2*np.pi)
        self.goal_pose = [goal_position_map[0], goal_position_map[1], random_yaw]
        self.publish_goal_pose()
        return self.goal_pose

    def publish_goal_pose(self):
        print(f"[{self.namespace}] Publishing goal pose: {self.goal_pose}, distance: \
              {np.linalg.norm(np.array(self.goal_pose)[:2] - np.array(self.robot_pose)[:2])}")
        goal_pose_stamped = self.create_pose_stamped(frame_id='map', pose=self.goal_pose)
        self.chosen_goals_list.poses.append(goal_pose_stamped.pose)
        self.chosen_goals_list.header.stamp = self.get_clock().now().to_msg()
        self.chosen_goals_list.header.frame_id = 'map'
        self.chosen_goals_list_pub.publish(self.chosen_goals_list)
        # self.goal_pose_stamped_pub.publish(goal_pose_stamped)

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
    
