
from .env import NavigationEnv
from .ros_interface import RosInterface
from stable_baselines3 import SAC
from pathlib import Path
import os
import rclpy
from ament_index_python.packages import get_package_share_directory


run_name = "balmy-sea-253"

package_name = "drl_navigation"
workspace_dir = Path(get_package_share_directory(package_name)).resolve().parents[3]
model_path = os.path.join(workspace_dir, "src", package_name, f'rl_models/best_model/{run_name}/')

def main():
    rclpy.init()
    ros_int = RosInterface()
    env = NavigationEnv(ros_interface=ros_int)


    model = SAC.load(f"{model_path}best_model.zip")
    params = model.get_parameters()
    model.set_parameters(params, exact_match=True)


    try:
        for i in range(10000):
            env.reset()
            for _ in range(1000):
                action, _states = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                if dones[0]:
                    env.reset()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        rclpy.shutdown()
        env.close()

if __name__ == "__main__":
    main()