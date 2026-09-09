import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import seaborn as sns
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import wandb


def plot_sample_trajectories(
    sample_episodes: list,
    patient: int,
    output_dir: str = "./patient_trajectory_plots",
    log_wandb: bool = False,
):
    """Plot multi-panel trajectory comparisons (Best, Median, Worst) for paper evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Patient {patient} — Representative Trajectories (Best, Median, Worst)",
        fontsize=16,
        fontweight="bold",
    )

    for ep in sample_episodes:
        steps = range(len(ep["agent_forces"]))
        suffix = ep.get("label_suffix", "")
        status = "Success" if ep["is_success"] else "Fail"
        label = f"Ep {ep['episode_id']}{suffix} ({status})"

        # 1. Position Error (mm)
        axes[0, 0].plot(steps, [p_val * 1000.0 for p_val in ep["pos_dists"]], label=label, linewidth=1.8)

        # 2. Angle Error (deg)
        axes[0, 1].plot(steps, [np.rad2deg(a_val) for a_val in ep["angle_dists"]], label=label, linewidth=1.8)

        # 3. Agent & Contact Forces (N)
        axes[1, 0].plot(steps, ep["agent_forces"], label=f"{label} Agent", linewidth=1.8)
        axes[1, 0].plot(steps, ep["contact_forces"], linestyle="--", alpha=0.6, label=f"{label} Contact")

        # 4. 2D Spatial Path (X-Y Plane in mm)
        if ep["positions"] and len(ep["positions"][0]) >= 2:
            pos_arr = np.array(ep["positions"]) * 1000.0  # m -> mm
            axes[1, 1].plot(pos_arr[:, 0], pos_arr[:, 1], marker="o", markersize=3, label=label, linewidth=1.5)

    # Panel Formatting & Limits
    axes[0, 0].axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Target Limit (0.5 mm)")
    axes[0, 0].set_title("Position Error over Time")
    axes[0, 0].set_ylabel("Error (mm)")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    axes[0, 1].axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Target Limit (0.5°)")
    axes[0, 1].set_title("Angle Error over Time")
    axes[0, 1].set_ylabel("Error (°)")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    axes[1, 0].axhline(0.4, color="red", linestyle="--", alpha=0.7, label="Force Limit (0.4 N)")
    axes[1, 0].set_title("Force Profile over Time")
    axes[1, 0].set_ylabel("Force (N)")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].legend(loc="upper right", fontsize=8)

    axes[1, 1].set_title("2D Spatial Path (X-Y Plane)")
    axes[1, 1].set_xlabel("X Position (mm)")
    axes[1, 1].set_ylabel("Y Position (mm)")
    axes[1, 1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"patient_{patient}_representative_trajectories.png")
    plt.savefig(plot_path, dpi=300)

    if log_wandb and wandb.run is not None:
        wandb.log({f"Plots/Patient_{patient}_Trajectories": wandb.Image(plot_path)})

    plt.close()
    print(f"Saved representative trajectory plot: {plot_path}")


def multiple_envs(
    model_path,
    patient=110,
    threshold_pos=0.0005,
    threshold_ori=0.00872665,  # np.deg2rad(0.5)
    maxforce=5,
    softtissue="spring",
    num_springs=3,
    youngs_modulus=1e7,
    vtk_file="rect0009.vtk",
    randomise_start=0,
    n_envs=1,
    num_eps=1000,
    log=0,
    seed=42,
    safemode=0,
    force_limit=0.4,
):
    threshold_ori = np.deg2rad(0.5)
    threshold_pos = 0.0005
    is_safemode = True if safemode == 1 else False

    goal_type = [0, 0, 0]
    env_kwargs = {
        "reward_type": "sparse",
        "max_steps": 100,
        "goal_type": goal_type,
        "horizon": "variable",
        "obs_type": "dict",
        "distance_threshold_pos": threshold_pos,
        "dt": 0.001,
        "dr": 0.01,
        "distance_threshold_ori": threshold_ori,
        "action_type": "euler",
        "start_pos": "home",
        "maxforce": maxforce,
        "contact_type": 1,
        "maximum_contact_force_threshold": force_limit,
        "number_of_springs": num_springs,
        "youngs_modulus_type": "testing",
        "randomise_num_springs": 1,
        "randomise_foot_dynamics": 1,
        "randomise_sensor_noise": 1,
        "randomise_start": randomise_start,
        "softtissue": softtissue,
        "patient": patient,
        "vtk_file": vtk_file,
        "safe_mode": is_safemode,
        "test": True,
        "render_mode": "direct",
    }

    env = make_vec_env(
        "gym_fracture:anklesurg-v2",
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv,
        seed=seed,
    )

    vec_norm_path = os.path.join(model_path, "vec_normalize.pkl")
    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found at: {vec_norm_path}")

    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False

    model_dir = Path(model_path)
    model_candidates = sorted(
        [p for p in model_dir.glob("model*") if p.is_file() and not p.name.endswith("-rb.pkl")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not model_candidates:
        raise FileNotFoundError(f"No model files starting with 'model' found in {model_dir}")

    selected_model = model_candidates[0]
    model = TD3.load(str(selected_model), env=env)

    # Global summary collections
    dones = []
    contacts = []
    position_error = []
    angle_error = []
    agent_force = []
    contact_forces = []
    interlock_triggers = []
    episode_steps = []
    force_breached = []
    contact_breached = []

    # Per-step tracking buffers
    ep_contact_forces = [[] for _ in range(env.num_envs)]
    ep_agent_forces = [[] for _ in range(env.num_envs)]
    ep_pos_dists = [[] for _ in range(env.num_envs)]
    ep_angle_dists = [[] for _ in range(env.num_envs)]
    ep_positions = [[] for _ in range(env.num_envs)]
    ep_orientations = [[] for _ in range(env.num_envs)]
    ep_step_counters = [0 for _ in range(env.num_envs)]

    # Storage for percentile sampling
    all_episodes_data = []
    episodes_collected = 0

    obs = env.reset()

    while episodes_collected < num_eps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones_array, info_list = env.step(action)

        for i in range(env.num_envs):
            ep_step_counters[i] += 1
            info_step = info_list[i]

            current_contact_force = info_step.get("contact_force", 0.0)
            current_agent_force = info_step.get("force", info_step.get("agent_force", 0.0))
            current_pos_dist = info_step.get("pos_distance", 0.0)
            current_angle_dist = info_step.get("angle", 0.0)
            current_pos = info_step.get("position", info_step.get("pos", None))
            current_rot = info_step.get("rotation", info_step.get("ori", None))

            ep_contact_forces[i].append(current_contact_force)
            ep_agent_forces[i].append(current_agent_force)
            ep_pos_dists[i].append(current_pos_dist)
            ep_angle_dists[i].append(current_angle_dist)

            if current_pos is not None:
                ep_positions[i].append(current_pos)
            if current_rot is not None:
                ep_orientations[i].append(current_rot)

            if dones_array[i]:
                info = info_list[i]

                is_success = info.get("is_success", False)
                has_contact = info.get("contact", False)
                max_force_val = info.get("maximum_force", info.get("force", 0.0))
                pos_dist = info.get("pos_distance", 0.0)
                angle_dist = info.get("angle", 0.0)
                interlocks = info.get("interlock_count", 0)
                steps_taken = ep_step_counters[i]

                peak_ep_contact = (
                    max(ep_contact_forces[i][0:]) if len(ep_contact_forces[i]) > 1 else 0.0
                )
                is_breached = 1 if max_force_val > maxforce else 0

                dones.append(is_success)
                contacts.append(has_contact)
                position_error.append(pos_dist)
                angle_error.append(angle_dist)
                agent_force.append(max_force_val)
                contact_forces.append(peak_ep_contact)
                interlock_triggers.append(interlocks)
                episode_steps.append(steps_taken)
                force_breached.append(is_breached)
                contact_breached.append(1 if peak_ep_contact > force_limit else 0)
                episodes_collected += 1

                # Save episode step trajectory for percentile sorting
                all_episodes_data.append({
                    "episode_id": episodes_collected,
                    "is_success": is_success,
                    "pos_dists": list(ep_pos_dists[i]),
                    "angle_dists": list(ep_angle_dists[i]),
                    "agent_forces": list(ep_agent_forces[i]),
                    "contact_forces": list(ep_contact_forces[i]),
                    "positions": list(ep_positions[i]),
                    "final_pos_error": pos_dist,
                })

                print(
                    f"[{episodes_collected}/{num_eps}] Patient {patient} | Env {i} "
                    f"Success: {is_success} | Peak Force: {max_force_val:.3f}N | "
                    f"Peak Contact: {peak_ep_contact:.3f}N  "
                    f"Interlocks: {interlocks} | Steps: {steps_taken} | "
                    f"Pos Err: {pos_dist:.5f}m | Angle Err: {np.rad2deg(angle_dist):.2f}° | "
                    f"Success Rate: {sum(dones) / len(dones):.2%}"
                )

                # Log step table to WandB
                if log == 1 and max_force_val <= 50:
                    traj_cols = [
                        "Step",
                        "Position_Distance_m",
                        "Angle_Distance_rad",
                        "Agent_Force_N",
                        "Contact_Force_N",
                    ]

                    has_pos_vec = len(ep_positions[i]) == len(ep_agent_forces[i])
                    has_rot_vec = len(ep_orientations[i]) == len(ep_agent_forces[i])

                    if has_pos_vec:
                        traj_cols.extend(["Pos_X", "Pos_Y", "Pos_Z"])
                    if has_rot_vec:
                        traj_cols.extend(["Rot_X", "Rot_Y", "Rot_Z"])

                    traj_table = wandb.Table(columns=traj_cols)

                    for step_idx in range(len(ep_agent_forces[i])):
                        row = [
                            step_idx,
                            ep_pos_dists[i][step_idx],
                            ep_angle_dists[i][step_idx],
                            ep_agent_forces[i][step_idx],
                            ep_contact_forces[i][step_idx],
                        ]
                        if has_pos_vec:
                            p_val = ep_positions[i][step_idx]
                            row.extend(p_val if hasattr(p_val, "__len__") else [p_val, 0, 0])
                        if has_rot_vec:
                            r_val = ep_orientations[i][step_idx]
                            row.extend(r_val if hasattr(r_val, "__len__") else [r_val, 0, 0])

                        traj_table.add_data(*row)

                    wandb.log(
                        {
                            "Patient_ID": patient,
                            "Episode": episodes_collected,
                            "Success": is_success,
                            "Contact": has_contact,
                            "Peak_Agent_Force": max_force_val,
                            "Force_Violation": is_breached,
                            "Peak_Contact_Force": peak_ep_contact,
                            "Contact_Violation": contact_breached[-1] if contact_breached else 0,
                            "Interlock_Triggers": interlocks,
                            "Episode_Steps": steps_taken,
                            "Position_Distance": pos_dist,
                            "Angle_Distance": angle_dist,
                            "Success_Rate": sum(dones) / len(dones),
                            f"Trajectories/Patient_{patient}_Ep_{episodes_collected}": traj_table,
                        }
                    )

                # Reset per-episode buffers
                ep_contact_forces[i] = []
                ep_agent_forces[i] = []
                ep_pos_dists[i] = []
                ep_angle_dists[i] = []
                ep_positions[i] = []
                ep_orientations[i] = []
                ep_step_counters[i] = 0

                if episodes_collected >= num_eps:
                    break

    # Percentile trajectory plot (Best, Median, Worst)
    if all_episodes_data:
        all_episodes_data.sort(key=lambda x: x["final_pos_error"])

        best_ep = all_episodes_data[0]
        median_ep = all_episodes_data[len(all_episodes_data) // 2]
        worst_ep = all_episodes_data[-1]

        best_ep["label_suffix"] = " (Best)"
        median_ep["label_suffix"] = " (Median)"
        worst_ep["label_suffix"] = " (Worst)"

        plot_sample_trajectories(
            sample_episodes=[best_ep, median_ep, worst_ep],
            patient=patient,
            output_dir="./patient_trajectory_plots",
            log_wandb=(log == 1),
        )

    # Save summary CSV
    model_name = Path(model_path).name
    df = pd.DataFrame(
        {
            "patient": patient,
            "success": dones,
            "contact": contacts,
            "pos_error_m": position_error,
            "angle_error_rad": angle_error,
            "peak_force_N": agent_force,
            "peak_contact_force_N": contact_forces,
            "force_violation": force_breached,
            "contact_violation": contact_breached,
            "interlock_triggers": interlock_triggers,
            "episode_steps": episode_steps,
        }
    )
    csv_filename = f"eval_patient_{patient}_{model_name}.csv"
    df.to_csv(csv_filename, index=False)

    # Print summary block
    success_no_contact = sum(1 for d_val, c_val in zip(dones, contacts) if d_val and not c_val)
    failure_no_contact = sum(1 for d_val, c_val in zip(dones, contacts) if not d_val and not c_val)
    success_contact = sum(1 for d_val, c_val in zip(dones, contacts) if d_val and c_val)
    failure_contact = sum(1 for d_val, c_val in zip(dones, contacts) if not d_val and c_val)

    print(f"\n================ Summary Table (Patient {patient}) ================")
    print(f"{'Metric':<35} {'Value':<15}")
    print("-" * 50)
    print(f"{'Overall Success Rate':<35} {np.mean(dones):.2%}")
    print(f"{'Force Violation Rate (>0.5N)':<35} {np.mean(force_breached):.2%}")
    print(f"{'Avg Interlock Triggers / Ep':<35} {np.mean(interlock_triggers):.2f}")
    print(f"{'Avg Steps to Completion':<35} {np.mean(episode_steps):.2f}")
    print(f"{'Avg Position Error (m)':<35} {np.mean(position_error):.6f}")
    print(f"{'Avg Angle Error (rad)':<35} {np.mean(angle_error):.6f}")
    print(f"{'Avg Peak Agent Force (N)':<35} {np.mean(agent_force):.4f}")
    print(f"{'Avg Contact Force (N)':<35} {np.mean(contact_forces):.4f}")
    print(f"{'Avg Contact Violation Rate':<35} {np.mean(contact_breached):.2%}")
    print("-" * 50)
    print(f"{'Success, No Contact':<35} {success_no_contact:<15}")
    print(f"{'Failure, No Contact':<35} {failure_no_contact:<15}")
    print(f"{'Success, Contact':<35} {success_contact:<15}")
    print(f"{'Failure, Contact':<35} {failure_contact:<15}")
    print("===================================================================\n")

    if log == 1:
        wandb.run.summary[f"patient_{patient}_success_rate"] = np.mean(dones)
        wandb.run.summary[f"patient_{patient}_violation_rate"] = np.mean(force_breached)
        wandb.run.summary[f"patient_{patient}_avg_interlocks"] = np.mean(interlock_triggers)
        wandb.run.summary[f"patient_{patient}_avg_steps"] = np.mean(episode_steps)
        wandb.run.summary[f"patient_{patient}_avg_pos_error"] = np.mean(position_error)
        wandb.run.summary[f"patient_{patient}_avg_angle_error"] = np.mean(angle_error)
        wandb.run.summary[f"patient_{patient}_avg_agent_force"] = np.mean(agent_force)
        wandb.run.summary[f"patient_{patient}_avg_contact_force"] = np.mean(contact_forces)
        wandb.run.summary[f"patient_{patient}_avg_contact_violation_rate"] = np.mean(contact_breached)
        wandb.run.summary[f"patient_{patient}_contact_violation"] = contact_breached
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model across randomized patients")
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="/media/catherine/Data/Best Models 30926/model-spring_randomYM_09031956_1",
        help="Path to trained model folder",
    )
    parser.add_argument("--maxforce", type=float, default=3.3, help="Max motor command force.")
    parser.add_argument("--safemode", type=int, default=0, help="Enable safe mode (1) or not (0).")
    parser.add_argument("--youngs_modulus", type=float, default=1e7, help="Tissue Young's modulus.")
    parser.add_argument("--randomise_start", type=int, default=1, help="Randomize start position (1) or not (0).")
    parser.add_argument("--force_limit", type=float, default=0.4, help="Force limit for violation detection (N)")
    parser.add_argument("--num_springs", type=int, default=3, help="Number of ligament springs.")
    parser.add_argument("--softtissue", type=str, default="spring", help="Soft Tissue Type.")
    parser.add_argument("--vtk_file", type=str, default="rect0009.vtk", help="VTK geometry file")
    parser.add_argument("--threshold_pos", type=float, default=0.0005, help="Position error limit (m)")
    parser.add_argument("--threshold_ori", type=float, default=0.5, help="Angle error limit (deg)")
    parser.add_argument("--n_envs", type=int, default=1, help="Parallel environment count")
    parser.add_argument("--num_eps", type=int, default=1000, help="Episodes to evaluate per patient")
    parser.add_argument("--log", type=int, default=0, help="Log to Weights & Biases (0 or 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.log == 1:
        model_name_clean = args.model_path.split("/")[-1].split(".")[0]
        if "random" in model_name_clean and args.safemode == 1:
            tags = ["random", "safe"]
        elif "random" in model_name_clean and args.safemode == 0:
            tags = ["random", "unsafe"]
        elif "random" not in model_name_clean and args.safemode == 1:
            tags = ["baseline", "safe"]
        else:
            tags = ["baseline", "unsafe"]

        wandb.init(project="Validation-results", name=f"Eval_{model_name_clean}", tags=tags)

    patients = [198, 102, 132, 252]
    for patient in patients:
        multiple_envs(
            model_path=args.model_path,
            patient=patient,
            threshold_pos=args.threshold_pos,
            threshold_ori=np.deg2rad(args.threshold_ori),
            maxforce=args.maxforce,
            safemode=args.safemode,
            softtissue=args.softtissue,
            num_springs=args.num_springs,
            youngs_modulus=args.youngs_modulus,
            randomise_start=args.randomise_start,
            vtk_file=args.vtk_file,
            n_envs=args.n_envs,
            num_eps=args.num_eps,
            log=args.log,
            seed=args.seed,
        )