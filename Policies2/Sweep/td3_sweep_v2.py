# td3_sweep_v2.py
import argparse
import datetime
from typing import Callable
import numpy as np
import gymnasium as gym
import wandb
from git import Repo, InvalidGitRepositoryError
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from success_callback import StopTrainingOnSuccessRate

repo_paths = [
    "/users/cop21cma/FracSoftGym/",
    "/home/catherine/FractureGym/",
    "/home/catherine/FractureSoftGym/",
]

def get_git_commit_hash(repo_path):
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha
    except Exception:
        return None

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def train():
    # 1. Initialize W&B run (W&B agent handles config injection automatically)
    run = wandb.init(project="Chapter3-Sweep", entity="cmabraham1-university-of-sheffield", sync_tensorboard=True, save_code=True)
    config = wandb.config

    action_type = "euler"
    threshold_pos = 0.0005
    threshold_ori = np.deg2rad(0.5)
    maxforce = 5
    softtissue = "spring"
    num_springs = 3
    contact_type = 1
    maximum_contact_force_threshold = 0.25
    youngs_modulus = 5e5
    eval_seed = 42
    seed = 1

    # Fetch hyperparameters from W&B sweep config
    learning_rate = config.learning_rate
    gamma = config.gamma
    tau = config.tau
    batch_size = config.batch_size
    train_freq = config.train_freq
    net_arch = config.net_arch
    learning_starts = config.learning_starts
    her_sampled_goal = config.her_sampled_goal
    action_noise_type = getattr(config, "action_noise", "normal")
    buffer_size = getattr(config, "buffer_size", 100000)

    env_kwargs = {
        "reward_type": "sparse",
        "max_steps": 100,
        "horizon": "variable",
        "obs_type": "dict",
        "distance_threshold_pos": threshold_pos,
        "dt": 0.001,
        "dr": 0.01,
        "distance_threshold_ori": threshold_ori,
        "action_type": action_type,
        "start_pos": "home",
        "patient": 110,
        "maxforce": maxforce,
        'maximum_contact_force_threshold': maximum_contact_force_threshold,
        "contact_type": contact_type,
        "number_of_springs": num_springs,
        "softtissue": softtissue,
        "test": False,
        "youngs_modulus": youngs_modulus,
        "render_mode": None,
    }

    env = make_vec_env("gym_fracture:anklesurg-v2", env_kwargs=env_kwargs, n_envs=1, vec_env_cls=DummyVecEnv, seed=seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    if action_noise_type == "normal":
        action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.1 * np.ones(env.action_space.shape[0]))
    elif action_noise_type == "OU":
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]), sigma=0.02 * np.ones(env.action_space.shape[0]))
    else:
        action_noise = None

    policy_kwargs = dict(net_arch=net_arch)

    model = TD3(
        policy="MultiInputPolicy",
        env=env,
        verbose=0,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(n_sampled_goal=her_sampled_goal),
        learning_rate=linear_schedule(learning_rate),
        train_freq=train_freq,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        gradient_steps=-1,
        seed=42,
        action_noise=action_noise,
        tensorboard_log=f"./logs/{run.id}",
    )

    eval_env_kwargs = env_kwargs.copy()
    eval_env = make_vec_env("gym_fracture:anklesurg-v2", env_kwargs=eval_env_kwargs, vec_env_cls=SubprocVecEnv, seed=eval_seed)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    success_callback = StopTrainingOnSuccessRate(vec_env=eval_env, max_no_improvement_evals=1, success_threshold=1)
    eval_callback = EvalCallback(eval_env, eval_freq=10000, deterministic=True, n_eval_episodes=20, callback_after_eval=success_callback)

    model.learn(250_000, callback=eval_callback)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, required=True, help="W&B Sweep ID")
    parser.add_argument("--count", type=int, default=5, help="Number of runs per agent")
    args = parser.parse_args()

    # Join the existing sweep
    wandb.agent(args.sweep_id, project="Chapter3-Sweep", entity="cmabraham1-university-of-sheffield", function=train, count=args.count)