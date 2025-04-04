import gymnasium as gym
import numpy as np
import time
from gymnasium import spaces

from tabulate import tabulate

from .ros_interface import RosInterface

MAX_EPISODE_STEPS = 1280
MAX_DISTANCE = 17.0  # depending on the .world dimensions (in this case 12x12)

MIN_LINEAR_VEL = 0.0
MAX_LINEAR_VEL = 0.22
MIN_ANGULAR_VEL = -1.00
MAX_ANGULAR_VEL = 1.00

DELTA_T = 0.1
COLLISION_THRESHOLD = 0.15
GOAL_THRESHOLD = 0.2


class NavigationEnv(gym.Env):
    def __init__(self, ros_interface=RosInterface) -> None:
        super().__init__()

        self.ros_interface = ros_interface

        self.action = [0.0, 0.0]
        self.goal_position = [0.0, 0.0]
        self.step_count = 0
        self.goal_min_distance = 4.5
        self.goal_max_distance = MAX_DISTANCE
        self.increase_goal_distance = False
        self.total_reward = 0.0

        self.lin_vel = 0.0
        self.ang_vel = 0.0


        self._init_obs_space()
        self._init_action_space()

    def _init_obs_space(self):
        # self.observation_space = spaces.Dict(
        #     {
        #         "scan_ranges": spaces.Box(
        #             low=0.0, high=1.0, shape=(360,), dtype=np.float32
        #         ),
        #         # distance to the goal, angle to the goal, linear vel, angular vel
        #         "robot_state": spaces.Box(
        #             low=0.0, high=1.0, shape=(4,), dtype=np.float32
        #         ),
        #     }
        # )
        self.observation_space = spaces.Dict(
            {
                "velocity": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
                "scan_ranges": spaces.Box(
                    low=-1.0, high=1.0, shape=(360,), dtype=np.float32
                ),
                "distance": spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "angle": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
            }
        )



    def _init_action_space(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def normalize(self, value, min_value, max_value):
        """Map `value` from `[min_value, max_value]` to `[0, 1]`"""
        return (value - min_value) / (max_value - min_value)

    def denormalize(self, value, min_value, max_value):
        """Map `value` from `[0, 1]` to `[min_value, max_value]`"""
        return value * (max_value - min_value) + min_value

    def normalize_symmetric(self, value, min_value, max_value):
        """Map `value` from `[min_value, max_value]` to `[-1, 1]`"""
        return 2 * ((value - min_value) / (max_value - min_value)) - 1

    def denormalize_symmetric(self, value, min_value, max_value):
        """Map `value` from `[-1, 1]` to `[min_value, max_value]`"""
        return ((value + 1) / 2) * (max_value - min_value) + min_value

    def _get_obs(self):
        collision = False
        goal_reached = False
        scan_ranges = self.ros_interface.get_scan_ranges()
        scan_min, scan_max = self.ros_interface.get_scan_min_max_range()
        scan_ranges = np.clip(scan_ranges, scan_min, scan_max)

        robot_position = self.ros_interface.get_odom_position()
        robot_orientation = self.ros_interface.get_odom_orientation()
        goal_distance = np.linalg.norm(self.goal_position - robot_position)
        goal_distance_vector = (self.goal_position - robot_position) / np.linalg.norm(
            self.goal_position - robot_position
        )
        robot_x_axis_vector = np.array(
            [np.cos(robot_orientation), np.sin(robot_orientation)]
        )
        # angle between the robot x axis and the goal distance vector (-pi to pi)
        goal_angle = np.arctan2(
            np.cross(robot_x_axis_vector, goal_distance_vector),
            np.dot(robot_x_axis_vector, goal_distance_vector),
        )


        if min(scan_ranges) < COLLISION_THRESHOLD:
            collision = True

        if goal_distance < GOAL_THRESHOLD:
            goal_reached = True



        self.lin_vel = self.ros_interface.get_odom_linear_velocity()[0]
        self.ang_vel = self.ros_interface.get_odom_angular_velocity()[2]

        linear_velocity = self.normalize_symmetric(self.lin_vel, MIN_LINEAR_VEL, MAX_LINEAR_VEL)
        angular_velocity = self.normalize_symmetric(self.ang_vel, MIN_ANGULAR_VEL, MAX_ANGULAR_VEL)

        velocity = np.array([linear_velocity, angular_velocity], dtype=np.float32)
        scan_ranges = np.array(
            self.normalize_symmetric(scan_ranges, scan_min, scan_max), dtype=np.float32
        )
        distance = np.array([self.normalize_symmetric(goal_distance, 0.0, MAX_DISTANCE)], dtype=np.float32)
        angle = np.array([np.cos(goal_angle), np.sin(goal_angle)], dtype=np.float32)

        # print(tabulate([["Goal distance", goal_distance],
        #                 ["Goal angle", goal_angle * 180 / np.pi],
        #                 ["distance norm", distance],
        #                 ["angle norm", angle]],
        #                 ["", ""], tablefmt="pretty"))


        observation = {"velocity": velocity,
                       "scan_ranges": scan_ranges,
                       "distance": distance,
                       "angle": angle}
                                             

        return observation, collision, goal_reached

    def _get_reward(self, observation, collision, goal_reached, action):
        # Optimized weights for 1280-step episodes
        GOAL_REWARD = 100.0            # Increased for clear success signal
        COLLISION_PENALTY = -50.0      # Reduced to prevent over-caution
        POS_LINEAR_VEL_REWARD = 0.1        # Encourage forward movement
        NEG_LINEAR_VEL_PENALTY = -0.1      # Discourage backward movement
        ANGULAR_VEL_PENALTY = -0.1     # Discourage unnecessary spinning
        DISTANCE_REWARD = 2.0          # Stronger incentive to approach goal
        OBSTACLE_PENALTY = -5.0        # Progressive penalty for obstacles
        TIME_PENALTY = -0.1            # Increased time pressure 

        self.lin_vel = self.ros_interface.get_odom_linear_velocity()[0]
        self.ang_vel = self.ros_interface.get_odom_angular_velocity()[2]

        lin_vel = self.normalize_symmetric(self.lin_vel, MIN_LINEAR_VEL, MAX_LINEAR_VEL)
        ang_vel = self.normalize_symmetric(self.ang_vel, MIN_ANGULAR_VEL, MAX_ANGULAR_VEL)

        # scan_min, scan_max = self.ros_interface.get_scan_min_max_range()
        # min_scan_range = self.denormalize(min(observation["scan_ranges"]), scan_min, scan_max)
        # goal_distance = self.denormalize(observation["robot_state"][0], 0.0, MAX_DISTANCE)

        # print(tabulate([["lin_vel", lin_vel],
        #                 ["ang_vel", ang_vel],
        #                 ["lin_reward = ", LINEAR_VEL_REWARD * max(lin_vel,0.0)],
        #                 ["ang_reward = ", ANGULAR_VEL_PENALTY * abs(ang_vel)]],
        #                 ["", ""], tablefmt="pretty"))
        
        # Reward components
        reward = 0.0
        reward += GOAL_REWARD * goal_reached
        reward += COLLISION_PENALTY * collision
        reward += POS_LINEAR_VEL_REWARD * max(lin_vel, 0.0)  
        # reward += NEG_LINEAR_VEL_PENALTY * abs(min(lin_vel, 0.0)) 
        reward += ANGULAR_VEL_PENALTY * abs(ang_vel)
        # reward += DISTANCE_REWARD / (1 + goal_distance)  # Inverse distance reward
        # reward += OBSTACLE_PENALTY * max(0, 1 - (min_scan_range/0.5))  # Progressive penalty <0.5m
        # reward += TIME_PENALTY * (self.step_count / MAX_EPISODE_STEPS)  # Scaled time penalty

        return reward

    def _get_terminated(self, collision, goal_reached):
        if collision:
            print("Collision")
        if goal_reached:
            print("Goal reached")

        return collision or goal_reached

    def _get_truncated(self):
        if self.step_count >= MAX_EPISODE_STEPS:
            print("Episode truncated")
            self.step_count = 0
            return True
        else:
            return False

    def _get_info(self, collision, goal_reached):
        return {"collision": collision, "goal_reached": goal_reached}

    def step(self, action):
        self.step_count += 1

        self.action = [
            self.denormalize_symmetric(action[0], MIN_LINEAR_VEL, MAX_LINEAR_VEL),
            self.denormalize_symmetric(action[1], MIN_ANGULAR_VEL, MAX_ANGULAR_VEL),
        ]

        # Pausing and unpausing the simulation makes the training faster,
        # and makes sure the robot isn't stuck during step cause of training calculations
        # (this actually happens every time the model is updated, in this case every 64 steps)

        self.ros_interface.unpause_simulation()
        self.ros_interface.publish_cmd_vel(self.action[0], self.action[1])
        start_time = self.ros_interface.get_gazebo_time()
        current_time = start_time
        while current_time - start_time < DELTA_T:
            current_time = self.ros_interface.get_gazebo_time()
            remaining = DELTA_T - (current_time - start_time)
            if remaining > 0:
                time.sleep(remaining)  # Free up CPU during wait
        self.ros_interface.pause_simulation()

        observation, collision, goal_reached = self._get_obs()
        reward = self._get_reward(
            observation, collision, goal_reached, self.action
        )  # be sure action is the scaled one

        self.total_reward += reward
        terminated = self._get_terminated(collision, goal_reached)
        truncated = self._get_truncated()
        info = self._get_info(collision, goal_reached)

        if goal_reached:
            self.increase_goal_distance = True
        # print(
        #     tabulate(
        #         [
        #             ["Reward", reward],
        #             ["Collision", collision],
        #             ["Goal reached", goal_reached],
        #             ["Action", action],
        #             ["Linear vel", self.action[0]],
        #             ["Angular vel", self.action[1]],
        #             ["Truncated", truncated],
        #             ["Terminated", terminated],
        #         ],
        #         ["", ""],
        #         tablefmt="pretty",
        #     )
        # )

        return observation, reward, terminated, truncated, info
    


    def check_pos(self, x, y) -> bool:
        map_obstacles = [
            [-5.3, -3.3, -1.3, 0.2],
            [-5.3, -4.2, 2.4, 3.5],
            [-5.0, -4.1, 4.0, 5.0],
            [-3.7, -1.0, -3.6, -1.5],
            [-2.5, -1.7, 0.0, 3.5],
            [-3.7, -0.5, 1.4, 2.2],
            [-0.5, 0.5, -4.5, -3.4],
            [-0.5, 0.5, 3.4, 4.5],
            [1.2, 3.6, -2.9, -2.0],
            [2.8, 3.6, -2.7, 0.3],
            [1.8, 3.7, 1.7, 3.2],
            [3.4, 4.5, 0.3, 1.5],
            [4.0, 5.5, -3.3, -4.4],
        ]

        # Check if outside map
        if not (-5.0 <= x <= 5.0 and -5.0 <= y <= 5.0):
            return False

        # Check if inside obstacle's bounding box
        for obstacle in map_obstacles:
            x1, x2, y1, y2 = obstacle
            if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
                return False
        return True

    def set_new_goal(self, center=[0.0, 0.0]) -> np.array:
        # slowly increase the range of the goal position
        if self.increase_goal_distance:
            self.increase_goal_distance = False
            if self.goal_max_distance < MAX_DISTANCE:
                self.goal_max_distance += 0.005
            if self.goal_min_distance < 3.0:
                self.goal_min_distance += 0.005

        goal_ok = False
        while not goal_ok:
            # Sample a random angle
            theta = np.random.uniform(0, 2 * np.pi)
            # Sample a random radius within the annular region
            r = np.random.uniform(self.goal_min_distance, self.goal_max_distance)
            # Convert polar coordinates to Cartesian
            x = center[0] + r * np.cos(theta)
            y = center[1] + r * np.sin(theta)
            # Validate the position
            goal_ok = self.check_pos(x, y)

        return np.array([x, y])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.total_reward = 0.0
        # Reset simulation
        self.ros_interface.reset_world()
        x_robot = 0
        y_robot = 0
        # Change robot position
        x_robot = 0
        y_robot = 0
        position_ok = False
        while not position_ok:
            x_robot = np.random.uniform(-5, 5)
            y_robot = np.random.uniform(-5, 5)
            position_ok = self.check_pos(x_robot, y_robot)
        self.ros_interface.set_entity_state(
            "burger", x_robot, y_robot, np.random.uniform(-np.pi, np.pi)
        )

        # Change goal
        self.goal_position = self.set_new_goal(center=[x_robot, y_robot])
        self.ros_interface.publish_goal_point(self.goal_position)
        print(f"Min goal distance: {self.goal_min_distance:.2f}, Max goal distance: {self.goal_max_distance:.2f}")

        # Change boxes position
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x, y = 0, 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = self.check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - x_robot, y - y_robot])
                distance_to_goal = np.linalg.norm([x - self.goal_position[0], y - self.goal_position[1]])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            
            self.ros_interface.set_entity_state(name, x, y, np.random.uniform(-np.pi, np.pi))




        observation, _, _ = self._get_obs()
        info = self._get_info(False, False)

        return observation, info

    def close(self) -> None:
        self.ros_interface.destroy_node()
