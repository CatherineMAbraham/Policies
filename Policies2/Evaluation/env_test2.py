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
                  vtk_file='rect005.vtk',
                  num_eps=100,
                  log=True,
                  seed=42):
        #Find the second last value in the model string 
        #contact_model= model_path.split('_')[-2]
        #print(f"Contact model from path: {contact_model}")
        # if contact_model == '0' :
        #         contact_type = 0
        # else:

        #         contact_type = 1
        #goal_pos = np.array([0.3091113269329071, -0.060230158269405365, 0.15660402178764343])
        #goal_ori = np.array([0.9997855687184534, -0.00617234947044787, -0.019766510735738874, -6.11364646429962e-05])
        #goal_type = np.concatenate((goal_pos, goal_ori))
        #threshold_pos = 0.001
        #threshold_ori = np.deg2rad(1)
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
                'number_of_springs': num_springs,
                'youngs_modulus': youngs_modulus,
                'vtk_file': vtk_file,
                'action_type': 'euler',
                'maxforce': maxforce,
                'contact_type' : 1,
                'start_pos' : 'home',
                'render_mode': 'human',
                'test': False,}
     
        # Remove . from beginning if present
        #model_path = model #"/media/catherine/Data/Chapter2Test1Results/Selected_Models/model-spring_3_5.0E+06_8_04080339"
        model_path2 = os.path.join("/home/catherine/Policies2/", model_path) 
        #print(f"Loading model from: {model_path2}")
        #model_path2 = os.path.join("/users/cop21cma/Policies2/TD3_Alg/", model_path)
        env = make_vec_env('gym_fracture:softsurg-v0', n_envs=n_envs, env_kwargs=env_kwargs,vec_env_cls=SubprocVecEnv, seed=seed)
        #model_path2 = os.path.join("/users/cop21cma/Policies2/TD3/", model_path)
        #model_path2= model #'/home/catherine/Policies2/Curriculum/model-spring_03190722'#"/home/catherine/Policies2/Evaluation/best_models/1/model-spring_03140755_1_0_1"
        env = VecNormalize.load(f"{model_path2}/vec_normalize.pkl", env) # Register the environment
        env.training = False
        env.norm_reward = False
        model_dir = Path(model_path2)
        model_candidates = sorted(
                [p for p in model_dir.glob("model*") if p.is_file() and not p.name.endswith("-rb.zip")],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
        )
        if not model_candidates:
                raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")

        selected_model = model_candidates[0]
        model = TD3.load(str(selected_model), env=env)
        #print(f"Loaded model: {selected_model}")
        dones = []
        contacts = []
        num = num_eps
        episodes_collected = 0
        obs = env.reset()
        #print(f"Initial observation: {obs}")
        eps = 0
        while episodes_collected < num:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones_array, info_list = env.step(action)
                
                for i in range(env.num_envs):
                        if dones_array[i]:
                                info = info_list[i]
                                
                                # 1. Get the actual final observation (before the auto-reset)
                                # This is critical if you want to calculate metrics manually
                                final_obs = info.get("terminal_observation")
                                
                                # 2. Get the success flag provided by the environment/Monitor
                                is_success = info.get("is_success", False)
                                
                                # 3. Get your custom 'contact' metric
                                # Note: Ensure your env puts 'contact' in the info dict even on the final step!
                                has_contact = info.get("contact", False)
                                
                                dones.append(is_success)
                                contacts.append(has_contact)
                                
                                episodes_collected += 1
                                print(f"[{episodes_collected}/{num}] Env {i} Success: {is_success} Force: {info.get('force')} Pos: {info.get('pos_distance')} Angle: {info.get('angle')} Contact: {has_contact}, Success Rate: {sum(dones) / len(dones)}")
                                ## If force >50 do not log to wandb as it is an outlier and can skew the results
                                # remove number of episodes collected from the success rate calculation in the log as well
                                if info.get('force', 0) <= 50:
                                       eps +=1
                                if log==1 :
                                    #table = wandb.Table(data = is_success,columns=["Episode", "Success"])
                                    #histogram = wandb.plot.Histogram(table,value='Success', title="Success Distribution")
                                    wandb.log({"Episode": episodes_collected,  "Contact": has_contact, "force": info.get('force', 0), "Position Distance": info.get('pos_distance', 0), "Angle Distance": info.get('angle', 0), "Success": is_success, "Success Rate": sum(dones) / len(dones)})
                                    if info.get('force', 0) <= 50:      
                                        wandb.run.summary["final_success_rate"] = sum(dones) / eps
                                if episodes_collected >= num:
                                        break
        success_no_contact = sum(1 for d, c in zip(dones, contacts) if d and not c)
        failure_no_contact = sum(1 for d, c in zip(dones, contacts) if not d and not c)
        success_contact = sum(1 for d, c in zip(dones, contacts) if d and c)
        failure_contact = sum(1 for d, c in zip(dones, contacts) if not d and c)

        print("\nSummary Table:")
        print(f"{'Outcome':<20} {'Count':<10}")
        print(f"{'Success, No Contact':<20} {success_no_contact:<10}")
        print(f"{'Failure, No Contact':<20} {failure_no_contact:<10}")
        print(f"{'Success, Contact':<20} {success_contact:<10}")
        print(f"{'Failure, Contact':<20} {failure_contact:<10}")
        print(f"\nOverall Success Rate: {sum(dones)/len(dones):.2%}")
        if log ==1:
                wandb.run.summary["overall_success_rate"] = sum(dones) / len(dones)
                wandb.run.summary["success_no_contact"] = success_no_contact
                wandb.run.summary["failure_no_contact"] = failure_no_contact
                wandb.run.summary["success_contact"] = success_contact
                wandb.run.summary["failure_contact"] = failure_contact
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the trained model on multiple environments")
    parser.add_argument("--model_path", type=str, help="Path to the trained model zip file")
    parser.add_argument('--maxforce', type=float, default=4, help='Force threshold for the environment.')
    parser.add_argument('--youngs_modulus', type=float, default=1e6, help='Young\'s modulus for the soft tissue.')
    parser.add_argument('--num_springs', type=int, default=3, help='Number of springs for the soft tissue.')
    parser.add_argument('--softtissue', type=str, default="soft", help='Soft Tissue Type.')
    parser.add_argument("--vtk_file", type=str, default=None, help="Path to the VTK file for soft tissue visualization")
    parser.add_argument("--threshold_pos", type=float, default=0.001, help="Position threshold for success")
    parser.add_argument("--threshold_ori", type=float, default=0.08, help="Orientation threshold for success")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments to test on")
    parser.add_argument("--num_eps", type=int, default=100, help="Number of episodes to collect data for")
    parser.add_argument("--log", type=int, default=0, help="Whether to log results to Weights & Biases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    #remove everything before model in the model path
    #model_name = args.model_path.split("/")[-1].split(".")[0]
    
    if args.log==1:
        wandb.init(project="softsurg", name=f"{args.model_path.split('/')[-1].split('.')[0]}")
    multiple_envs(
    model_path=args.model_path,
    maxforce=args.maxforce,
    num_springs=args.num_springs,
    softtissue=args.softtissue,
    vtk_file=args.vtk_file,
    youngs_modulus=args.youngs_modulus,
    threshold_pos=args.threshold_pos,
    threshold_ori=args.threshold_ori,
    n_envs=args.n_envs,
    num_eps=args.num_eps,
    log=args.log,
    seed=args.seed
)
