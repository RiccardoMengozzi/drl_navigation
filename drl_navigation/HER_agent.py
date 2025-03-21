import rclpy
import threading
import wandb
import os
import numpy as np

from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from wandb.integration.sb3 import WandbCallback   
from rclpy.executors import MultiThreadedExecutor

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from .HER_env import HerNavigationEnv
from .ros_interface import RosInterface


TOTAL_TIMESTEPS = 256000
MAX_EPISODE_STEPS = 640
LOAD_MODEL = False
    
class CustomLoggingCallback(BaseCallback):
    def __init__(self, env: HerNavigationEnv, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.env = env
        self.total_steps = 0
        self.total_episodes = 0
        self.current_episode_length = 0
        self.current_episode_rewards = []
        # For episode-based metrics
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        self.total_steps += 1
        self.current_episode_length += 1
        
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]  # Assuming single environment
        done = self.locals["dones"][0]
        truncated = info.get("TimeLimit.truncated", False)

        # Store current episode rewards
        self.current_episode_rewards.append(reward)

        # Calculate current total reward
        current_total_reward = sum(self.current_episode_rewards)

        # Dynamic terminal logging (updates same line)
        print(f"Ep {self.total_episodes + 1} | "
              f"Steps: {self.current_episode_length} | "
              f"Current Reward: {current_total_reward:.2f} | "
              f"Running Total: {self.total_steps}", 
              end='\r', flush=True)
        
        # Log step-based metrics
        wandb.log({
            "step/reward": reward,
            "step/total_steps": self.total_steps,
            "step/lin_vel": self.env.get_wrapper_attr("lin_vel"),
            "step/ang_vel": self.env.get_wrapper_attr("ang_vel"),
        }, commit=False)

        if done or truncated:
            self.total_episodes += 1
            episode_reward = sum(self.current_episode_rewards)
            
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            # Log episode-based metrics
            wandb.log({
                "episode/reward": episode_reward,
                "episode/length": self.current_episode_length,
                "episode/avg_reward": episode_reward / self.current_episode_length,
                "episode/total_episodes": self.total_episodes,
            }, commit=False)

            # Reset episode tracking
            self.current_episode_length = 0
            self.current_episode_rewards = []

        # Commit all logged metrics
        wandb.log({}, commit=True)
        
        return True

def main() -> None:
    try:
        package_name = "drl_navigation"
        workspace_dir = Path(get_package_share_directory(package_name)).resolve().parents[3]
        log_path = os.path.join(workspace_dir, "src", package_name, 'rl_logs/')

        rclpy.init()
        ros_int = RosInterface()
        executor = MultiThreadedExecutor()
        executor.add_node(ros_int)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()

        wandb_config = {
                    "entity": "RiccardoMengozzi",
                    "policy_type": "MultiInputPolicy",
                    "total_timesteps": TOTAL_TIMESTEPS,
                    }
        
        if LOAD_MODEL:
            # Load the previous run ID
            with open(f"{log_path}HER_wandb_id.txt", "r") as f:
                run_id = f.read().strip()

            # Resume the wandb run
            run = wandb.init(
                id=run_id,
                resume="must",  # or "must" to enforce resuming
                project="drl_navigation",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                mode='online'  
            )
        else:
            run = wandb.init(
                project="drl_navigation",
                config=wandb_config,
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                mode='online'  
            )
            with open(f"{log_path}HER_wandb_id.txt", "w") as f:
                f.write(run.id)

        checkpoints_path = os.path.join(workspace_dir, "src", package_name, f'rl_models/checkpoints/HER_{run.name}/')
        best_model_path = os.path.join(workspace_dir, "src", package_name, f'rl_models/best_model/HER_{run.name}/')
            
        wand_cb = WandbCallback(gradient_save_freq=1000,
                                verbose=2)
        
        env = HerNavigationEnv(ros_interface=ros_int)
        env = Monitor(env)


        # Configure action noise (if needed)
        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(env.action_space.shape),
            sigma=0.2 * np.ones(env.action_space.shape),
            theta=0.15,      
        )

        custom_cb = CustomLoggingCallback(env)

        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=TOTAL_TIMESTEPS / 20,
            save_path=checkpoints_path,
            name_prefix=run.name
        )

        eval_callback = EvalCallback(
            env,
            best_model_save_path=best_model_path,
            log_path=log_path,
            eval_freq=TOTAL_TIMESTEPS / 10,
            deterministic=True,
            render=False,

        )

        replay_buffer_kwargs = dict(
            n_sampled_goal=4,             # Number of virtual goals per transition
            goal_selection_strategy="future",    # Best for navigation
            copy_info_dict=True,          # Copy all the information in the transition dict
        )

        callbacks = [wand_cb, checkpoint_callback, eval_callback, custom_cb]

        if LOAD_MODEL:
            model = SAC.load(f"{best_model_path}best_model.zip")
            params = model.get_parameters()
            new_model = SAC(
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
                tensorboard_log=log_path,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=replay_buffer_kwargs,
                learning_starts=MAX_EPISODE_STEPS, # Delayed start for HER
                verbose=1,
                device="auto",               # Uses GPU if available
            )
            new_model.set_parameters(params, exact_match=True)
            new_model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
        else:
            model = SAC(
                'MultiInputPolicy',
                env,
                learning_rate=0.001,          # Linear decay can also be used
                buffer_size=1_000_000,       # Replay buffer size
                batch_size=256,              # Mini-batch size
                tau=0.005,                   # Target network update rate
                gamma=0.95,                  # Discount factor
                train_freq=64,               # Update model every N steps
                gradient_steps=64,           # Number of gradient steps per update
                # action_noise=action_noise,   # Only if needed for exploration
                ent_coef='auto',             # Entropy coefficient
                stats_window_size=MAX_EPISODE_STEPS // 10,
                tensorboard_log=log_path,
                replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=replay_buffer_kwargs,
                learning_starts=MAX_EPISODE_STEPS * 2, # Delayed start for HER
                verbose=1,
                device="auto",               # Uses GPU if available
            )
            
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, log_interval=1)
        print("Learning finished")

    except KeyboardInterrupt:
        print("Shutting down")
    
    finally:
        env.close()
        run.finish()
        rclpy.shutdown()




if __name__ == '__main__':  
    main()