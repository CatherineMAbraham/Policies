import os

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
                  seed=42,vtk_file=None):
        #Find the second last value in the model string 
        #contact_model= model_path.split('_')[-2]
        #print(f"Contact model from path: {contact_model}")
        # if contact_model == '0' :
        #         contact_type = 0
        # else:

        #         contact_type = 1
        env_kwargs = {
                'reward_type': 'sparse',
                'max_steps': 100,
                'horizon': 'variable',
                'obs_type': 'dict',
                'distance_threshold_pos': threshold_pos,
                'dt': 0.001,
                'dr':0.01,
                'distance_threshold_ori': threshold_ori,
                'softtissue': softtissue,
                'vtk_file': vtk_file,
                'number_of_springs': num_springs,
                'youngs_modulus': youngs_modulus,
                'action_type': 'euler',
                'maxforce': maxforce,
                'contact_type' : 0,
                'start_pos' : 'home',
                'render_mode': None,
                'test': True,}
        # Remove . from beginning if present
        model_path = "./model-spring_3_5.0E+06_8_04080339"
        #model_path2 = os.path.join("/users/cop21cma/Policies2/TD3_Alg/", model_path)
        env = make_vec_env('gym_fracture:softsurg-v0', env_kwargs=env_kwargs,vec_env_cls=SubprocVecEnv, seed=seed)
        #model_path2 = os.path.join("/users/cop21cma/Policies2/TD3/", model_path)
        #model_path2= model #'/home/catherine/Policies2/Curriculum/model-spring_03190722'#"/home/catherine/Policies2/Evaluation/best_models/1/model-spring_03140755_1_0_1"
        #env = VecNormalize.load(f"{model_path}/vec_normalize.pkl", env) # Register the environment
        env.training = False
        env.norm_reward = False

        # model_dir = Path(model_path)
        # model_candidates = sorted(
        #         [p for p in model_dir.glob("model*") if p.is_file() and not p.name.endswith("-rb.zip")],
        #         key=lambda p: p.stat().st_mtime,
        #         reverse=True,
        # )
        # if not model_candidates:
        #         raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")

        # selected_model = model_candidates[0]
        # #model = TD3.load(str(selected_model), env=env)
        #print(f"Loaded model: {selected_model}")
        dones = []
        contacts = []
        force = []
        force_axis = []
        num = 100
        episodes_collected = 0
        obs = env.reset()
        #print(f"Initial observation: {obs}")
        eps = 0
        ep_force_values = [[] for _ in range(n_envs)]
        # instead of using the model to take actions, we will define a path for the robot to follow and record the forces along the way.
        #randomly sample actions from the action space 

        while episodes_collected < num:
                        action = env.action_space.sample()
                        action = np.asarray(action)
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
                                if log == 1:
                                        force_axis_mean = step_info.get("force_axis_mean", [0, 0, 0])
                                        wandb.log({
                                                "Step Force": step_info.get("force", 0),
                                                "X Force": force_axis_mean[0],
                                                "Y Force": force_axis_mean[1],
                                                "Z Force": force_axis_mean[2],
                                                "Position Distance": step_info.get("pos_distance", 0),
                                                "Angle Distance": step_info.get("angle", 0),
                                        })
                                
                                if not done_flag:
                                        continue

                                is_success = step_info.get("is_success", False)
                                has_contact = step_info.get("contact", False)
                                valid_force_values = ep_force_values[env_idx]
                                force_value = float(np.mean(valid_force_values)) if valid_force_values else np.nan
                                dones.append(is_success)
                                contacts.append(has_contact)
                                if np.isfinite(force_value):
                                        force.append(force_value)
                                episodes_collected += 1
                                ep_force_values[env_idx] = []
                                if np.isfinite(force_value) and force_value <= 50:
                                        eps += 1
                                
                                # Log episode summary
                                if log == 1:
                                        wandb.log({
                                                "Episode": episodes_collected,
                                                "Contact": has_contact,
                                                "Episode Force Mean": force_value,
                                                "Success": is_success,
                                                "Success Rate": sum(dones) / len(dones),
                                        })
                                        if np.isfinite(force_value) and force_value <= 50 and eps > 0:
                                                wandb.run.summary["final_success_rate"] = sum(dones) / eps

                                if episodes_collected >= num:
                                        break
        print(force)
        print(np.mean(force) if force else np.nan)
        print(np.max(force) if force else np.nan)
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
        # success_no_contact = sum(1 for d, c in zip(dones, contacts) if d and not c)
        # failure_no_contact = sum(1 for d, c in zip(dones, contacts) if not d and not c)
        # success_contact = sum(1 for d, c in zip(dones, contacts) if d and c)
        # failure_contact = sum(1 for d, c in zip(dones, contacts) if not d and c)

        # print("\nSummary Table:")
        # print(f"{'Outcome':<20} {'Count':<10}")
        # print(f"{'Success, No Contact':<20} {success_no_contact:<10}")
        # print(f"{'Failure, No Contact':<20} {failure_no_contact:<10}")
        # print(f"{'Success, Contact':<20} {success_contact:<10}")
        # print(f"{'Failure, Contact':<20} {failure_contact:<10}")
        # print(f"\nOverall Success Rate: {sum(dones)/len(dones):.2%}")
        # if log ==1:
        #         wandb.run.summary["overall_success_rate"] = sum(dones) / len(dones)
        #         wandb.run.summary["success_no_contact"] = success_no_contact
        #         wandb.run.summary["failure_no_contact"] = failure_no_contact
        #         wandb.run.summary["success_contact"] = success_contact
        #         wandb.run.summary["failure_contact"] = failure_contact
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained model on multiple environments")
    parser.add_argument("--model_path", type=str, help="Path to the trained model zip file")
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
    args = parser.parse_args()
    #remove everything before model in the model path
    #model_name = args.model_path.split("/")[-1].split(".")[0]
    
    if args.log==1:
        wandb.init(project="meshconvergence", name=f"{args.softtissue}_{args.num_springs}_youngs_{args.youngs_modulus}")
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
    seed=args.seed,vtk_file=args.vtk_file
)
 