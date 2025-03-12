import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .ros_interface import ROSInterface
import time
import rclpy


ACTION_PERIOD = 0.2
MAX_LINEAR_VELOCITY = 0.22
MAX_ANGULAR_VELOCITY = 1.5

MAX_GOAL_DISTANCE = 2.0

COLLISION_THRESHOLD = 0.12
GOAL_DISTANCE_TOLERANCE = 0.2
GOAL_ANGLE_TOLERANCE = 0.1 # ~5°





class NavigationEnv(gym.Env):
    def __init__(self, ros_interface: ROSInterface, max_episode_steps = 1000) -> None:
        super().__init__()
        self.ros_int = ros_interface
        self.namespace = self.ros_int.namespace

        self.max_episode_steps = max_episode_steps

        self.goal_pose = np.array([0.0, 0.0, 0.0])
        self.total_angle = 0.0
        self.last_time = 0.0
        self.step_count = 0
        self.is_resetting = False        
        self.reset_future = None


        # Be sure all the callbacks have been called at least once
        self.wait_for_first_msgs(['tf_map2odom', 'tf_odom2foot','gazebo_clock_msg', 'env_properties', 'odom_msg', 'reset_gazebo_msg'])
        # Be sure the robot pose is not None
        self.ros_int.get_robot_pose()

        

        self._init_obs_space()
        self._init_action_space()



    def wait_for_first_msgs(self, msgs):
        last_print_time = 0.0
        while any(getattr(self.ros_int, msg) is None for msg in msgs):
            # Check if 1 second has passed since the last print
            current_time = time.time()
            if current_time - last_print_time >= 1.0:
                last_print_time = current_time  # Update last print time
                for msg in msgs:
                    if getattr(self.ros_int, msg) is None:
                        print(f"[{self.namespace}] Waiting for {msg}")
            pass
        print(f"[{self.namespace}] All messages received")


    def _init_obs_space(self):
        self.observation_space = spaces.Dict({
            'scan': spaces.Box(low=-1, high=1, shape=(360,), dtype=np.float32),
            'goal_rel_pose': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),            
        })

    def _init_action_space(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


    def normalize(self, data, min, max):
        data = np.clip(data, min, max)
        return 2*((data - min) / (max - min)) - 1
    

    def detect_collision(self, scan_ranges):
        collision = False
        for scan in scan_ranges:
            if scan < COLLISION_THRESHOLD:
                collision = True
                break
        return collision
    
    def is_goal_reached(self, distance, angle_diff):
        # env_cheker only works with python-bools (not numpy)   
        return bool(distance < GOAL_DISTANCE_TOLERANCE and angle_diff < GOAL_ANGLE_TOLERANCE)
    
    def get_total_angle(self):
        angular_velocity = self.ros_int.get_angular_velocity()
        dt = self.ros_int.get_gazebo_time() - self.last_time
        self.last_time = self.ros_int.get_gazebo_time()
        self.total_angle += abs(angular_velocity) * dt
        return self.total_angle
    
    def is_gazebo_resetting(self):
        return self.ros_int.get_reset_gazebo_msg()


    def _get_obs(self):
        scan_ranges = np.array(self.ros_int.get_scan_ranges())
        robot_pose = np.array(self.ros_int.get_robot_pose())

        scan_ranges_bounds = self.ros_int.get_scan_bounds()
        scan_ranges = self.normalize(scan_ranges, scan_ranges_bounds[0], scan_ranges_bounds[1])

        # if self.namespace == 'env_0': print(f"[{self.namespace}] robot_pose: {self.ros_int.get_robot_pose()}")

        goal_rel_pose = self.goal_pose - robot_pose
        goal_rel_pose = [self.normalize(goal_rel_pose[0], -MAX_GOAL_DISTANCE, MAX_GOAL_DISTANCE), \
                         self.normalize(goal_rel_pose[1], -MAX_GOAL_DISTANCE, MAX_GOAL_DISTANCE), \
                         self.normalize(goal_rel_pose[2], -np.pi/2, np.pi/2)]
        
        # print(f"[{self.namespace}] goal_rel_pose: {goal_rel_pose}")

        return {
            'scan': np.array(scan_ranges, dtype=np.float32),
            'goal_rel_pose': np.array(goal_rel_pose, dtype=np.float32),
        }
    
    def total_angle_reward(self, x, a, k):
        # sigmoid
        return 1/(1+np.exp(-a*(x+k)))
    
    def angle_diff(self, angle1, angle2):
        return np.abs((angle1 - angle2 + np.pi) % (2 * np.pi) - np.pi) # ensures minimum angular difference
    
    def distance(self, pose1, pose2):
        return np.linalg.norm(pose1[:2] - pose2[:2])
    
    def _get_reward(self):
        scan_ranges = self.ros_int.get_scan_ranges()
        robot_pose = self.ros_int.get_robot_pose()
        goal_pose = self.goal_pose

        W1, W2, W3, W4, W5 = -1, -1/np.pi*5, -5, -10, 20

        distance = self.distance(goal_pose, robot_pose)
        angle_diff = self.angle_diff(goal_pose[2], robot_pose[2])
        total_angle = self.get_total_angle()
        collision = self.detect_collision(scan_ranges)
        goal_reached = self.is_goal_reached(distance, angle_diff)

        r1 = W1*distance
        r2 = W2*angle_diff
        r3 = W3*self.total_angle_reward(total_angle, a = 0.6, k=-8)
        r4 = W4*collision
        r5 = W5*goal_reached

        total_reward = r1 + r2 + r3 + r4 + r5
        #rewards intervals
        #distance: [-MAX_DISTANCE, 0]
        #angle_diff: [-5, 0]
        #total_angle: [-5 (slow until 4-5 radians of total angle, fast after), 0]
        #collision: [-10, 0]
        #goal_reached: [0, 20]
        
#         print(f"""
# {'=' * 50}
# Environment: {self.namespace}, step: {self.step_count}
# {'=' * 50}
# {'Metric':<15}{'Value':<10}{'Weight':<10}{'Weighted Contribution'}
# {'-' * 50}
# {'Distance':<15}{distance:<10.2f}{W1:<10}{r1:.2f}
# {'Angle Diff':<15}{angle_diff:<10.2f}{W2:<10.2f}{r2:.2f}
# {'Total Angle':<15}{total_angle:<10.2f}{r3/total_angle:<10.2f}{r3:.2f}
# {'Collision':<15}{collision:<10}{W4:<10}{r4:.2f}
# {'Goal Reached':<15}{goal_reached:<10}{W5:<10}{r5:.2f}
# {'=' * 50}
# {'Total Reward':<35}{total_reward:.2f}
# {'=' * 50}
# """)

        return total_reward

    def _get_terminated(self):
        scan_ranges = self.ros_int.get_scan_ranges()
        robot_pose = self.ros_int.get_robot_pose()
        goal_pose = self.goal_pose

        distance = self.distance(goal_pose, robot_pose)
        angle_diff = self.angle_diff(goal_pose[2], robot_pose[2])
        # print(f"[{self.namespace}] Goal yaw = {goal_pose[2]:.2f}, Robot yaw = {robot_pose[2]:.2f}")

        collision = self.detect_collision(scan_ranges)
        goal_reached = self.is_goal_reached(distance, angle_diff)
        # print(f"[{self.namespace}] Collision: {collision}, Goal reached: {goal_reached}")

        return bool(collision or goal_reached) # env_cheker only works with python-bools (not numpy)   

    def _get_truncated(self, step_count):
        if step_count >= self.max_episode_steps:
            print(f"[{self.namespace}] Episode truncated")
            self.step_count = 0
            return True
        return False

    def _get_info(self):
        return {}

    def scale_action(self, action):
        self.linear = action[0] * MAX_LINEAR_VELOCITY
        self.angular = action[1] * MAX_ANGULAR_VELOCITY
        return [self.linear, self.angular]

    def wait_action_period(self, period):
        start_time = self.ros_int.get_gazebo_time()
        while self.ros_int.get_gazebo_time() - start_time < period:
            pass



    def step(self, action):
        self.step_count = self.step_count + 1
        if self.is_gazebo_resetting():
            # while the reset is not done, return null observations
            self.wait_action_period(ACTION_PERIOD)
            observation = {
                'scan': np.zeros(360, dtype=np.float32),
                'goal_rel_pose': np.zeros(3, dtype=np.float32),
            }
            reward = 0.0
            terminated = False
            truncated = True
            info = {}    
            return observation, reward, terminated, truncated, info


        # if self.namespace == 'env_0' and self.step_count == 10:
        #     self.ros_int.reset_environment()
        if self.reset_future is not None:
            if self.reset_future.done() and self.is_resetting:
                if self.reset_future.result().success:
                    print(f"[{self.namespace}] Reset done")
                    self.goal_pose = self.ros_int.set_random_goal_pose(max_distance=MAX_GOAL_DISTANCE)
                    self.is_resetting = False



        if not self.is_resetting:
            scaled_action = self.scale_action(action)
            # start timer for angular velocity integration just before the first action is published
            if self.last_time == 0.0:
                self.last_time = self.ros_int.get_gazebo_time()
            self.ros_int.publish_action(scaled_action)
            self.wait_action_period(ACTION_PERIOD)

            observation = self._get_obs()
            reward = self._get_reward()
            terminated = self._get_terminated()
            truncated = self._get_truncated(self.step_count)
            info = self._get_info()
        else:
            # while the reset is not done, return null observations
            self.wait_action_period(ACTION_PERIOD)
            observation = {
                'scan': np.zeros(360, dtype=np.float32),
                'goal_rel_pose': np.zeros(3, dtype=np.float32),
            }
            reward = 0.0
            terminated = False
            truncated = False
            info = {}    

        

        return observation, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        while self.is_gazebo_resetting():
            pass
        if not self.is_resetting:
            self.totol_angle = 0.0 # reset total angle
            self.reset_future = self.ros_int.reset_environment() # reset environment
            self.is_resetting = True
            # return null observations
            observation = {
                'scan': np.zeros(360, dtype=np.float32),
                'goal_rel_pose': np.zeros(3, dtype=np.float32),
            }
            info = {}
        else:
            observation = self._get_obs()
            info = self._get_info()
        return observation, info

    def close(self) -> None:
        self.ros_int.node.destroy_node()