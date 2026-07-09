import os
from git import Repo
from git import InvalidGitRepositoryError
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
import wandb
import argparse
from pathlib import Path

repo_paths = ["/users/cop21cma/FracSoftGym", "/home/catherine/FractureGym", '/home/catherineabraham/FractureSoftGym/']

def get_git_commit_hash(repo_path):
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        print(f"Invalid Git repository at {repo_path}")
    except Exception as e:
        print(f"An error occurred while getting the commit hash: {e}")
        return None

def multiple_envs(model_path,
                  threshold_pos=0.001, 
                  threshold_ori=0.08,
                  maxforce=50,
                  softtissue='soft',
                  youngs_modulus=1e7,
                  num_springs=3,
                  num_envs=1,
                  num_eps=100,
                  log=True,
                  seed=42,
                  vtk_file=None,
                  experts=['trajectory_5']): # Expecting a list of expert names
        
        render_mode = None
        for repo_path in repo_paths:
                try:
                        commit = get_git_commit_hash(repo_path)
                        if commit is not None:
                                print(f"Git commit hash for repository at {repo_path}: {commit}")
                                if repo_path in ["/users/cop21cma/FracSoftGym", "/home/catherineabraham/FractureSoftGym/"]:
                                        render_mode = None
                                        log = True 
                                break
                except Exception as e: 
                        print(f"Could not get commit hash for repository at {repo_path}: {e}")
        
        experiment_action = {}
        episode_length = []
        
        # Load all expert actions into dictionary
        for expert in experts:
                trajectory_path = f"./experts2/{expert}.npz"
                with np.load(trajectory_path, allow_pickle=True) as data:
                        if 'acts' not in data:
                                raise KeyError(f"Expected 'acts' in trajectory file {trajectory_path}")
                        experiment_action[expert] = np.asarray(data['acts'])
                        episode_length.append(experiment_action[expert].shape[0])
                if experiment_action[expert].ndim == 1:
                        experiment_action[expert] = experiment_action[expert][:, None]

        if vtk_file == 'None' or vtk_file is None:
                softtissue = None
        else:
                softtissue = 'soft'
                
        env_kwargs_list = []
        for env_idx in range(num_envs):
                expert_name = experts[env_idx]
                single_ep_length = experiment_action[expert_name].shape[0] # Extracted integer
                
                individual_kwargs = {
                        'reward_type': 'sparse',
                        'max_steps': single_ep_length, # <-- Now a single clean integer per env!
                        'horizon': 'variable',
                        'obs_type': 'dict',
                        'distance_threshold_pos': threshold_pos,
                        'dr': 1,
                        'dt': 1,
                        'action_type': 'euler',
                        'distance_threshold_ori': threshold_ori,
                        'start_pos': 'home',
                        'render_mode': render_mode,
                        'softtissue': softtissue,
                        'vtk_file': vtk_file,
                        'number_of_springs': num_springs,
                        'youngs_modulus': youngs_modulus,
                        'maxforce': maxforce,
                        'contact_type': 0,
                        'test': True
                }
                env_kwargs_list.append(individual_kwargs)
        
        # 2. Pass the list of dictionaries directly using env_kwargs
        env = make_vec_env(
                'gym_fracture:anklesurg-v1', 
                env_kwargs=None, # Clear standard global dict field
                vec_env_cls=DummyVecEnv, 
                num_envs=num_envs, 
                seed=seed,
                wrapper_class=None,
                env_kwargs_list=env_kwargs_list # Pass your custom per-env lists here
        )
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        
        force_axis = []
        env_current_step = np.zeros(num_envs, dtype=int)
        ep_force_values = [[] for _ in range(num_envs)]
        step_force = []
        
        # We will run until all environments have successfully completed their full trajectories
        # without being tripped by excessive force safety failures
        env_complete = np.zeros(num_envs, dtype=bool)
        
        obs = env.reset()
        print(f"Starting parallel replay with on-the-fly force-fail resetting across {num_envs} slots.")

        # Main execution control loop
        while not np.all(env_complete):
                parallel_actions = []
                
                # Gather the correct action step for each environment based on its personal index
                for env_idx in range(num_envs):
                        expert_name = experts[env_idx]
                        step_idx = env_current_step[env_idx]
                        
                        # Safety ceiling catch if an environment already successfully finished
                        if step_idx >= experiment_action[expert_name].shape[0]:
                                step_idx = experiment_action[expert_name].shape[0] - 1
                                env_complete[env_idx] = True
                                
                        current_act = experiment_action[expert_name][step_idx]
                        
                        action_pos = current_act[0:3]
                        action_ori = np.deg2rad(current_act[3:6])
                        action = np.concatenate([action_pos, action_ori])
                        parallel_actions.append(action)
                
                # Stack to correct matrix format: shape (num_envs, 6)
                actions_matrix = np.vstack(parallel_actions)
                
                # Step all parallel environments simultaneously
                obs, reward, done, info = env.step(actions_matrix)

                done_array = np.asarray(done).reshape(-1)
                info_list = list(info) if isinstance(info, (list, tuple)) else [info]

                # Parse metrics and check safety constraints for each track
                for env_idx, done_flag in enumerate(done_array):
                        if env_complete[env_idx]:
                                continue # Skip finished tracks
                                
                        step_info = info_list[env_idx]
                        if isinstance(step_info, tuple) and len(step_info) == 1 and isinstance(step_info[0], dict):
                                step_info = step_info[0]
                                
                        ep_force_value = step_info.get("force", 0)
                        
                        # --- THE RESET AND PURGE LOGIC ---
                        if step_info.get("truncated") and step_info.get("force_fail", False):
                                print(f"[CRITICAL FORCE] Env {env_idx} hit {ep_force_value:.2f}N. Resetting timeline and clearing historical log.")
                                
                                # 1. Completely wipe the dirty force log accumulated for this env slot so far
                                ep_force_values[env_idx] = []
                                if env_idx == 0:
                                        step_force = [] # Clear global tracking container if tracking slot 0
                                        force_axis = []
                                
                                # 2. Reset its tracking step back to the starting line
                                env_current_step[env_idx] = 0
                                
                                # Stable-Baselines3 vector environments automatically call env.reset() 
                                # internally for this specific sub-process on this frame.
                                continue 
                        
                        # If it didn't fail, append the data normally
                        if not np.isnan(ep_force_value) and 0 < ep_force_value < maxforce:
                                ep_force_values[env_idx].append(ep_force_value)
                        
                        # Remove the env_idx == 0 gate for step_force, just log to wandb from slot 0
                        if env_idx == 0 and log and ep_force_value < 100:
                                force_axis_mean = step_info.get("force_axis_mean", [0, 0, 0])
                                wandb.log({"Step Force": ep_force_value})

                        # Progress track
                        env_current_step[env_idx] += 1
                        if env_current_step[env_idx] >= experiment_action[experts[env_idx]].shape[0]:
                                env_complete[env_idx] = True

        try:
                p.disconnect()
        except Exception:
                pass

        # Return the clean, isolated history arrays for ALL running environments
        return ep_force_values

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Test the trained model on multiple environments")
        parser.add_argument("--model_path", type=str, help="Path to the trained model zip file")
        parser.add_argument("--trajectory_path", type=str, default="/home/catherine/Policies/Policies1/Test/trajectory_2024-02-23-10-42-50-All_data.npz", help="Path to the expert trajectory NPZ file")
        parser.add_argument('--maxforce', type=float, default=4, help='Force threshold for the environment.')
        parser.add_argument('--youngs_modulus', type=float, default=1e7, help='Young\'s modulus for the soft tissue.')
        parser.add_argument('--num_springs', type=int, default=3, help='Number of springs for the soft tissue.')
        parser.add_argument('--softtissue', type=str, default="soft", help='Soft Tissue Type.')
        parser.add_argument("--threshold_pos", type=float, default=0.001, help="Position threshold for success")
        parser.add_argument("--threshold_ori", type=float, default=0.08, help="Orientation threshold for success")
        parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments to test on")
        parser.add_argument("--num_eps", type=int, default=100, help="Number of episodes to collect data for")
        parser.add_argument("--log", type=int, default=0, help="Whether to log results to Weights & Biases")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
        parser.add_argument("--vtk_file", type=str, default=None, help="Path to the VTK file for soft tissue visualization")
        parser.add_argument("--expert", type=str, default="trajectory_1", help="Expert trajectory to use or list of comma-separated trajectories")
        args = parser.parse_args()

        # Parse comma separated experts list or build list manually based on n_envs
        if "," in args.expert:
                expert_list = [e.strip() for e in args.expert.split(",")]
        else:
                # Fallback if a single expert string is passed: replicate it to fill the env slots
                expert_list = [args.expert for _ in range(args.n_envs)]

        if args.log == 1:
                wandb.init(project="meshconvergence", name=f"{args.vtk_file}_{args.expert}_{args.youngs_modulus}", tags=[args.expert, 'model_trajectory_y'])
        
        multiple_envs(
                model_path=args.model_path,
                maxforce=args.maxforce,
                num_springs=args.num_springs,
                softtissue=args.softtissue,
                youngs_modulus=args.youngs_modulus,
                threshold_pos=args.threshold_pos,
                threshold_ori=args.threshold_ori,
                num_envs=args.n_envs,
                num_eps=args.num_eps,
                log=bool(args.log),
                seed=args.seed,
                vtk_file=args.vtk_file,
                experts=expert_list,
        )