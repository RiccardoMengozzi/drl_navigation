import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .ros_interface import ROSInterface
import time
import rclpy


ACTION_PERIOD = 0.2
MAX_LINEAR_VELOCITY = 0.22
MAX_ANGULAR_VELOCITY = 1.5

MIN_SCAN_RANGE = 0.12
MAX_SCAN_RANGE = 3.5

MAX_GOAL_DISTANCE = 5.0



class NavigationEnv(gym.Env):
    def __init__(self, namespace: str, ros_interface: ROSInterface):
        super().__init__()
        self.namespace = namespace
        self.ros_int = ros_interface

        self.goal_pose = np.array([0.0, 0.0, 0.0])

        self.wait_for_first_msgs(['scan_msg', 'tf_map2odom', 'tf_odom2foot','gazebo_clock_msg', 'env_properties'])
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


    def _init_obs_space(self):
        self.observation_space = spaces.Dict({
            'scan': spaces.Box(low=0, high=1, shape=(360,), dtype=np.float32),
            'goal_rel_pose': spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32),            
        })

    def _init_action_space(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)


    def normalize(self, data, min, max):
        return (data - min) / (max - min)


    def _get_obs(self):
        scan_ranges = np.array(self.ros_int.get_scan_ranges())
        scan_ranges = self.normalize(scan_ranges, MIN_SCAN_RANGE, MAX_SCAN_RANGE)
        robot_pose = np.array(self.ros_int.get_robot_pose())

        # if self.namespace == 'env_0': print(f"[{self.namespace}] robot_pose: {self.ros_int.get_robot_pose()}")

        goal_rel_pose = self.goal_pose - robot_pose
        goal_rel_pose = [self.normalize(goal_rel_pose[0], -MAX_GOAL_DISTANCE, MAX_GOAL_DISTANCE), \
                         self.normalize(goal_rel_pose[1], -MAX_GOAL_DISTANCE, MAX_GOAL_DISTANCE), \
                         self.normalize(goal_rel_pose[2], -np.pi/2, np.pi/2)]
        
        # print(f"[{self.namespace}] goal_rel_pose: {goal_rel_pose}")

        return {
            'scan': scan_ranges,
            'goal_rel_pose': goal_rel_pose,
        }

    def _get_reward(self, observation):
        pass

    def _get_done(self):
        pass

    def _get_truncated(self, observation):
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
        scaled_action = self.scale_action(action)
        self.ros_int.publish_action(scaled_action)
        self.wait_action_period(ACTION_PERIOD)


        observation = self._get_obs()
        reward = self._get_reward(observation)
        done = self._get_done()
        truncated = self._get_truncated(observation)
        info = self._get_info()



        return observation, reward, done, truncated, info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.goal_pose = self.ros_int.set_random_goal_pose()
        # print(f"[{self.namespace}] New goal pose: {self.goal_pose}")
        obs = self._get_obs()
        info = self._get_info()
        return obs, info
