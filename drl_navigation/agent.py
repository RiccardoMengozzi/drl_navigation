import rclpy
import os
import yaml
import numpy as np

from ament_index_python.packages import get_package_share_directory

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from wandb.integration.sb3 import WandbCallback 

from .navigation_vec_env import NavigationEnv
from .ros_interface import ROSInterface



MAX_EPISODE_STEPS = 500
TOTAL_TIMESTEPS = 100000

class CustomLoggingCallback(BaseCallback):
    def __init__(self, envs: SubprocVecEnv, num_envs, verbose=0):
        super(CustomLoggingCallback, self).__init__(verbose)
        self.envs = envs
        self.n_envs = num_envs
        self.step = 0 # Synchronized between all environments
        self.episodes = np.array([0 for _ in range(self.n_envs)])
        self.steps_per_episode = np.array([0 for _ in range(self.n_envs)])
        self.rewards = np.array([0 for _ in range(self.n_envs)])
        self.cumulated_reward = np.array([0 for _ in range(self.n_envs)])
        self.mean_episode_reward = np.array([0 for _ in range(self.n_envs)])
        self.done = np.array([False for _ in range(self.n_envs)])
        self.truncated = np.array([False for _ in range(self.n_envs)])


        self.rows = []

    def _on_step(self) -> bool:
        self.step += 1
        self.steps_per_episode = np.array([steps + 1 for steps in self.steps_per_episode])
        self.rewards = np.array(self.locals["rewards"])
        self.cumulated_reward = np.array([cumulated_reward + reward for cumulated_reward, reward in zip(self.cumulated_reward, self.rewards)])

        # print(f"Step: {self.step}")
        # print(f"Rewards: {self.rewards}")
        # print(f"Cumulated reward: {self.cumulated_reward}")
        # print(f"Steps per episode: {self.steps_per_episode}")

        dones = np.array(self.locals["dones"])
        truncated = np.array([self.locals["infos"][i].get("TimeLimit.truncated") for i in range(self.n_envs)])
        infos = np.array(self.locals["infos"])

        for i in range(self.n_envs):
            if dones[i] or truncated[i]:
                self.episodes[i] += 1
                self.mean_episode_reward[i] = self.cumulated_reward[i] / self.steps_per_episode[i]

                print(f"mean_episode_reward: {self.mean_episode_reward[i]}")
                self.cumulated_reward[i] = 0
                self.steps_per_episode[i] = 0

        # Log the interactive W&B plot
        wandb.log({
            **{f'mean_episode_reward/env_{i}' : reward for i, reward in enumerate(self.mean_episode_reward)},
            **{f'steps_per_episode/env_{i}' : steps for i, steps in enumerate(self.steps_per_episode)},
            **{f"episodes/env_{i}": count for i, count in enumerate(self.episodes)},
            "global_step": self.step
        })

        return True



def load_yaml(file_path):
    """
    Load a YAML file and return its content as a dictionary.

    :param file_path: Path to the YAML file.
    :return: Parsed YAML content as a dictionary.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)



def make_env(rank):
    def _init():
        return NavigationEnv(ROSInterface(f'env_{rank}'), max_episode_steps=MAX_EPISODE_STEPS)
    return _init

def main():
    print("Starting...")
    ########################### ENVIRONMENT CHECKER #################################
    ########## (check_env needs to be used with a single environment) ###############
    # env = create_env(0 , max_episode_steps=MAX_EPISODE_STEPS)
    # check_env(env)
    # rclpy.shutdown()
    #################################################################################


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
    # env = SubprocVecEnv([lambda i=i: create_env(i, max_episode_steps=MAX_EPISODE_STEPS) for i in range(num_envs)], start_method='spawn')
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)]) 
    # vec_env = Monitor(vec_env)

    wandb_config = {
                "entity": "RiccardoMengozzi",
                "policy_type": "MultiInputPolicy",
                "total_timesteps": TOTAL_TIMESTEPS,
                "learning_rate":1e-3,
                }
    
    run = wandb.init(
            project="drl_navigation",
            config=wandb_config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            mode='online'  
    )
        
    wand_cb = WandbCallback(gradient_save_freq=100,
                            verbose=2)
    
    # Initialize the PPO model with the parallel environments
    model = PPO("MultiInputPolicy", vec_env,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                clip_range=0.1,
                ent_coef=0.1,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=True,  # Important for continuous action spaces
                sde_sample_freq=4,
                verbose=1)

    callbacks = [wand_cb, CustomLoggingCallback(vec_env, num_envs)]

    # Train the model for the specified number of timesteps
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval = 1, callback=callbacks)

    # Shutdown rclpy after training
    rclpy.shutdown()

if __name__ == "__main__":
    main()
