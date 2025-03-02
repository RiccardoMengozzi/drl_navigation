import rclpy
import os
import yaml
import threading

from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from ament_index_python.packages import get_package_share_directory

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.utils import set_random_seed

from .navigation_vec_env import NavigationEnv
from .ros_interface import ROSInterface


def load_yaml(file_path):
    """
    Load a YAML file and return its content as a dictionary.

    :param file_path: Path to the YAML file.
    :return: Parsed YAML content as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def create_env(rank, seed=0):
    """
    Utility function for creating a multiprocessed custom environment.

    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    """
    rclpy.init()  # Initialize rclpy for this process
    namespace = f'env_{rank}'
    
    # Create the ROS interface for this environment
    ros_int = ROSInterface(namespace)
    executor = SingleThreadedExecutor()
    executor.add_node(ros_int)
    # # Create a separate thread to handle the ROS interface
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start() 
    
    # Create and return the environment
    env = NavigationEnv(namespace, ros_interface=ros_int)
    # set_random_seed(seed)
    # env.seed(seed + rank)
    return env



def main():

    yaml_file_path = os.path.join(
        get_package_share_directory('tb3_multi_env_spawner'),
        '..','..','..','..',
        'src',
        'tb3_multi_env_spawner',
        'config',
        'launch_params.yaml'
    )

    # Load launch parameters from YAML
    params = load_yaml(yaml_file_path)

    # Extract parameters from the YAML file
    num_envs = params['env']['num_envs']
    
    # Create a SubprocVecEnv with multiple environments running in parallel
    env = SubprocVecEnv([lambda i=i: create_env(i) for i in range(num_envs)])

    import numpy as np
    while True:
        env.reset()
        # action = np.array([np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)])
        action = np.array([0.0,0.0])
    # check_env(env)
    
    # Initialize the PPO model with the parallel environments
    model = PPO("MultiInputPolicy", env, verbose=1)
    
    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=10000)

    # Shutdown rclpy after training
    rclpy.shutdown()

if __name__ == "__main__":
    main()
