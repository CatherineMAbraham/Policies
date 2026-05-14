import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import wandb
import numpy as np
from typing import Callable
import datetime
from git import Repo, InvalidGitRepositoryError
import argparse
import log_callback
from success_callback import StopTrainingOnSuccessRate
import matplotlib.pyplot as plt
import time
threshold_pos = 0.001
threshold_ori = 5
action_type = 'euler'
maxforce = 3.5
softtissue = 'soft'
num_springs = 10
contact_type = 0
youngs_modulus = 1e6
env_kwargs = {
        'reward_type': 'sparse',
        'max_steps': 100,
        'horizon': 'variable',
        'obs_type': 'dict',
        'distance_threshold_pos': threshold_pos,
        'dt': 0.001,
        'dr':0.01,
        'distance_threshold_ori': threshold_ori,
        'action_type': action_type,
        'start_pos' : 'home',
        'maxforce': maxforce,
        'contact_type' :contact_type,
        'number_of_springs':num_springs,
        'softtissue':softtissue,
        'test': False,
        'youngs_modulus': youngs_modulus,
        'render_mode': 'human'}
        #"0.025 -0.04 0" rpy="0 1.57 0"
    
#env = make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs, n_envs=1,vec_env_cls=SubprocVecEnv)
#env = VecNormalize(env, norm_obs=True, norm_reward=False)
env = gym.make('gym_fracture:softsurg-v0', **env_kwargs)
action = [0,1,0,0,0,0]
obs,_ = env.reset()
forces = []
stretch = []
for i in range(30):
    obs, reward, done,truncated, info = env.step(action)
    forces.append(info['force'])
    stretch.append(info['stretch'])
    print(i)
    #time.sleep(0.1)
plt.plot(stretch, forces)
plt.xlabel('Stretch (m)')
plt.ylabel('Force (N)')
plt.title(f'Force vs Stretch for {softtissue} with Young\'s Modulus {youngs_modulus} Pa')
plt.grid()
plt.show()