import os

from git import Repo, Repo
from git import InvalidGitRepositoryError
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import wandb
import argparse
from pathlib import Path
repo_paths = ["/users/cop21cma/FracSoftGym", "/home/catherine/FractureGym",'/home/catherineabraham/FractureSoftGym/']
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
                  maxforce=5,
                  softtissue='soft',
                  youngs_modulus=1e7,
                  num_springs=3,
                  n_envs=1,
                  num_eps=100,
                  log=True,
                  seed=42,
                  vtk_file=None,
                  expert='trajectory_5'):
        "Testing different VTK files, Young's Modulus Values against the expert trajectories, recording forces along the way and plotting them against time."
        render_mode = None
        #render_mode = 'human'
        #log = 0 
        for repo_path in repo_paths:
                try:
                        commit = get_git_commit_hash(repo_path)
                        if commit is not None:
                                print(f"Git commit hash for repository at {repo_path}: {commit}")
                                if repo_path == "/users/cop21cma/FracSoftGym" or repo_path == "/home/catherineabraham/FractureSoftGym/":
                                        render_mode = None
                                        log =1 
                                break
                except Exception as e: print(f"Could not get commit hash for repository at {repo_path}: {e}")
        current_dir = os.getcwd()
        #trajectory_path = os.path.join(current_dir, f"/Experts/{expert}.npz")
        trajectory_path =  f"./Experts/{expert}.npz"
        with np.load(trajectory_path, allow_pickle=True) as expert:
                if 'acts' not in expert:
                        raise KeyError(f"Expected 'acts' in trajectory file {trajectory_path}")
                experiment_action = np.asarray(expert['acts'])

        if experiment_action.ndim == 1:
                experiment_action = experiment_action[:, None]

        #print(experiment_action)
        episode_length = experiment_action.shape[0]
        start = [ 0.35701957, -0.06,        0.15526956, 1.80000000e+02, 1.57154150e-13, 2.28130818e-02]
        delta = [0.0125,0.008,0.003,15,5,15]
        goal_pos =  np.array([start[0]+delta[0], start[1]-delta[1], start[2]+delta[2]])
        goal_ori = p.getQuaternionFromEuler(np.deg2rad([start[3]+delta[3], start[4]+delta[4], start[5]+delta[5]+0]))
        
        goal = np.concatenate([goal_pos, goal_ori])
        print(f"Goal: {goal}")
        if vtk_file =='None':
                softtissue = None
        else:
                softtissue = 'soft'
        env_kwargs = {
                'reward_type': 'sparse',
                'max_steps': episode_length,
                'horizon': 'variable',
                'obs_type': 'dict',
                'distance_threshold_pos': threshold_pos,
                'dr':0.01,
                'dt': 0.001,
                'action_type': 'euler',
                'distance_threshold_ori': threshold_ori,
                'start_pos' : 'home',
                'render_mode': render_mode,
                'softtissue': softtissue,
                'vtk_file': vtk_file,
                'number_of_springs': num_springs,
                'youngs_modulus': youngs_modulus,
                'maxforce': maxforce,
                'contact_type' : 0,
                'test': True}
        
        env = make_vec_env('gym_fracture:anklesurg-v1', env_kwargs=env_kwargs,vec_env_cls=DummyVecEnv, seed=seed)
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
        current_dir = os.getcwd()
        #model_path= f"{current_dir}/model-None_0_1.0_03220908/model-None_0_1.0_03220908"
        #model = TD3.load(model_path, env=env)
        force = []
        force_axis = []
        num = 1
        episodes_collected = 0
        obs = env.reset()
        
        eps = 0
        ep_force_values = [[] for _ in range(n_envs)]
        # instead of using the model to take actions, we will define a path for the robot to follow and record the forces along the way.
        # Replay the actions saved in the expert trajectory file.
        stop_replay = False
        #while episodes_collected < num:
        complete = False
        step_force =[] 
        action = np.array([0,0.6,0,0,0,0,0])
        while complete == False:
                i = 0
                restart_from_zero = False
                while i < len(experiment_action):
                        #print(episode_length)
                        action_pos = experiment_action[i][0:3]/1000
                        action_ori = experiment_action[i][3:6]
                        # action_ori= p.getQuaternionFromEuler(np.deg2rad(action_ori))
                        #print(f"Step {i+1}/{episode_length}, Action: {action}")
                        action = np.concatenate([action_pos, action_ori])#action#model.predict(obs)[0]  # 
                        if action.ndim == 1:
                                action = action[None, :]
                        #print(action)
                        obs, reward, done, info = env.step(action)

                        done_array = np.asarray(done).reshape(-1)
                        info_list = list(info) if isinstance(info, (list, tuple)) else [info]

                        for env_idx, done_flag in enumerate(done_array):
                                step_info = info_list[env_idx]
                                if isinstance(step_info, tuple) and len(step_info) == 1 and isinstance(step_info[0], dict):
                                        step_info = step_info[0]
                                ep_force_value = step_info.get("force", 0)
                                if not np.isnan(ep_force_value) and ep_force_value < 50 and ep_force_value > 0:
                                        ep_force_values[env_idx].append(ep_force_value)
                                force_axis.append(step_info.get("force_axis_mean"))

                                # Log all steps
                                #log = 0 
                                if log == 1 and step_info.get("force", 0)<100:
                                        force_axis_mean = step_info.get("force_axis_mean", [0, 0, 0])
                                        #print(step_info.get("force", 0))
                                        wandb.log({
                                                "Step Force": step_info.get("force", 0),
                                                "X Force": force_axis_mean[0],
                                                "Y Force": force_axis_mean[1],
                                                "Z Force": force_axis_mean[2],
                                                "Position Distance": step_info.get("pos_distance", 0),
                                                "Angle Distance": step_info.get("angle", 0),
                                        })

                                step_force.append(step_info.get("force", 0))
                                if not done_flag:
                                        i+=1
                                        continue

                                if step_info.get("truncated") and step_info.get("force_fail", 0) == True:
                                        print("Restarting replay from action 0 after excessive force termination.")
                                        ## I want to remove the logging and start a new wandb run for this new episode, so that the data is not mixed with the previous one.
                                        # if log == 1:
                                        #         wandb.finish()
                                        #         ##delete the run from wandb, so that it does not show up in the dashboard.
                                        #         run = wandb.run
                                        #         if run is not None:
                                        #                 run_id = run.id
                                        #                 api = wandb.Api()
                                        #                 api.delete_run(f"meshconvergence/{run_id}")
                                        #         wandb.init(project="meshconvergence", name=f"{vtk_file}_{expert}_{youngs_modulus}_restart",tags=[expert,'model_trajectory_y'])
                                        obs = env.reset()
                                        restart_from_zero = True
                                        i=0
                                        break
                                
                                if restart_from_zero == True:
                                        break
                        if restart_from_zero == True:
                                continue
                if restart_from_zero == False:
                        complete = True
        # print(force)
        # print(np.mean(force) if force else np.nan)
        # print(np.max(force) if force else np.nan)
        # Plot x, y, z force components against time for valid axis samples.
        valid_force_axis = []
        for axis_vec in force_axis:
                if axis_vec is None:
                        continue
                axis_arr = np.asarray(axis_vec, dtype=float).reshape(-1)
                if axis_arr.size >= 3 and np.all(np.isfinite(axis_arr[:3])):
                        valid_force_axis.append(axis_arr[:3])

        if valid_force_axis:
                axis_data = np.vstack(valid_force_axis)
                dt = env_kwargs.get('dt', 1.0)
                t = np.arange(axis_data.shape[0]) * dt

                fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
                labels = ["Fx", "Fy", "Fz"]
                for i in range(3):
                        axes[i].plot(t, axis_data[:, i], linewidth=1.2)
                        axes[i].set_ylabel(labels[i])
                        axes[i].grid(True, alpha=0.3)

                axes[-1].set_xlabel("Time (s)")
                fig.suptitle("Force Axis Components vs Time")
                plt.tight_layout()
                #plt.show()
                plt.savefig("./force_axis_components.png")
        else:
                print("No valid force_axis_mean data to plot.")
        return step_force
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
        parser.add_argument("--expert", type=str, default="trajectory_1", help="Expert trajectory to use")
        args = parser.parse_args()

        if args.log == 1:
                wandb.init(project="meshconvergence", name=f"{args.vtk_file}_{args.expert}_{args.youngs_modulus}",tags=[args.expert,'model_trajectory_y'])
        
        multiple_envs(
                model_path=args.model_path,
                maxforce=args.maxforce,
                num_springs=args.num_springs,
                softtissue=args.softtissue,
                youngs_modulus=args.youngs_modulus,
                threshold_pos=args.threshold_pos,
                threshold_ori=args.threshold_ori,
                n_envs=args.n_envs,
                num_eps=args.num_eps,
                log=args.log,
                seed=args.seed,
                vtk_file=args.vtk_file,
                expert=args.expert,
        )
 