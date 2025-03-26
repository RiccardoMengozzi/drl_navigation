
from .env import NavigationEnv
from .ros_interface import RosInterface
from stable_baselines3 import SAC
from pathlib import Path
import os
import rclpy
import numpy as np
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
import threading
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise


EPISODES = 100
MAX_STEPS = 100

run_name = "balmy-sea-253"

package_name = "drl_navigation"
workspace_dir = Path(get_package_share_directory(package_name)).resolve().parents[3]
model_path = os.path.join(workspace_dir, "src", package_name, f'rl_models/best_model/{run_name}/')

def main():
    rclpy.init()
    ros_int = RosInterface()
    executor = MultiThreadedExecutor()
    executor.add_node(ros_int)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    env = NavigationEnv(ros_interface=ros_int)

    # Configure action noise (if needed)
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(env.action_space.shape),
        sigma=0.2 * np.ones(env.action_space.shape),
        theta=0.15,      
    )

    test_model = SAC(
        'MultiInputPolicy',
        env,
        learning_rate=0.001,          # Linear decay can also be used
        buffer_size=1_000_000,       # Replay buffer size
        batch_size=256,              # Mini-batch size
        tau=0.005,                   # Target network update rate
        gamma=0.95,                  # Discount factor
        train_freq=64,               # Update model every N steps
        gradient_steps=64,           # Number of gradient steps per update
        action_noise=action_noise,   # Only if needed for exploration
        ent_coef='auto',             # Entropy coefficient
        stats_window_size=10,
        verbose=1,
        device="auto",               # Uses GPU if available
        policy_kwargs=dict(
            net_arch=[256, 256],
            log_std_init=-2        # (exp(-2): moderate exploration noise)
        )
    )

    trained_model = SAC.load(f"{model_path}best_model.zip")
    params = trained_model.get_parameters()
    test_model.set_parameters(params, exact_match=True)

    success = 0
    try:
        for i in range(EPISODES):
            obs, info = env.reset()
            for _ in range(MAX_STEPS):
                action, _states = test_model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                if done or truncated:
                    if info["collision"]:
                        print(f"Episode {i} failed: COLLISION")
                    elif info["goal_reached"]:
                        success += 1
                        print(f"Episode {i} success: GOAL REACHED")
                    elif truncated:
                        print(f"Episode {i} failed: TRUNCATED")
                    obs, info = env.reset()
                    break
        print("Success rate: ", success/EPISODES)


    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        rclpy.shutdown()
        env.close()

if __name__ == "__main__":
    main()