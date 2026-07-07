from git import InvalidGitRepositoryError, Repo
import numpy as np
import pandas as pd
import glob
import os
from force_from_expert import multiple_envs
import scipy.optimize as opt
import wandb
from scipy.signal import savgol_filter
## For each young's modulus, run the 5 trajectories using the VTK file we have picked during mesh convergence testing. 
# Then we want to work out the average trajectory on a 1-100% complete basis 
# Work out the rmse and use an optimiser to find the best young's modulus for the average trajectory.
def get_git_commit_hash(repo_path):
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        return repo.head.commit.hexsha
    except InvalidGitRepositoryError:
        print(f"Invalid Git repository at {repo_path}")
    except Exception as e:
        print(f"An error occurred while getting the commit hash: {e}")
        return None
def clean_force_spikes_rolling(data: np.ndarray, window: int = 7, threshold: float = 3.0) -> np.ndarray:
    """
    Detects and removes simulation explosions/spikes using a localized rolling Z-score
    instead of a global dataset mean, preventing unphysical flatlines.
    """
    series = pd.Series(data)
    
    # Calculate localized rolling metrics
    rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window, center=True, min_periods=1).std().fillna(0)
    
    # Prevent division by zero in perfectly flat regions
    rolling_std[rolling_std == 0] = 1e-6
    
    # Identify anomalies locally
    z_scores = (series - rolling_mean) / rolling_std
    is_spike = np.abs(z_scores) > threshold
    
    # Wipe spikes and smoothly interpolate over gaps
    clean_series = series.copy()
    clean_series[is_spike] = np.nan
    return clean_series.interpolate(method='linear').ffill().bfill().values
norm_time = np.linspace(0, 100, 101)


def normalize_force_trajectory(force_values: np.ndarray, completion_grid: np.ndarray = norm_time) -> np.ndarray:
    force_values = np.asarray(force_values, dtype=float).reshape(-1)
    if force_values.size == 0:
        return np.full(completion_grid.shape, np.nan)

    clean_force = savgol_filter(force_values, window_length=15, polyorder=2)#clean_force_spikes_rolling(force_values)
    original_completion = np.linspace(0, 100, clean_force.size)
    return np.interp(completion_grid, original_completion, clean_force)

# def get_expert():
#     file_pattern = './Sanjeev/*.csv'
#     file_list = sorted(glob.glob(file_pattern))
#     all_normalized_csvs = []

#     for file in file_list:
#         try:
#             df_raw = pd.read_csv(file)
#             suffix = os.path.splitext(os.path.basename(file))[0]
            
#             cols_to_keep = ['.fX.data', '.fY.data', '.fZ.data']
#             for col in cols_to_keep:
#                 df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
            
#             df_clean = df_raw[cols_to_keep].dropna()
            
#             f_x = df_clean['.fX.data'].values
#             f_y = df_clean['.fY.data'].values
#             f_z = df_clean['.fZ.data'].values
#             resultant = np.linalg.norm(np.column_stack([f_x, f_y, f_z]), axis=1)
            
#             original_indices = np.linspace(0, 100, len(resultant))
#             res_normalized = np.interp(norm_time, original_indices, resultant)
            
#             all_normalized_csvs.append(pd.Series(res_normalized, name=f'CSV_{suffix}'))
            
#         except Exception as e:
#             print(f"Error processing local CSV file {file}: {e}")

#     if all_normalized_csvs:
#         csv_df = pd.concat(all_normalized_csvs, axis=1)
#         csv_mean = csv_df.mean(axis=1).values
#         csv_std = csv_df.std(axis=1).values
#         ##Impulse for experimental data (area under the curve)
#         impulse_csv = np.trapezoid(csv_mean, norm_time)
#         print(f'Impulse for Experimental Reference: {impulse_csv:.2f}')
#     else:
#         csv_df = pd.DataFrame()
#         print("Error: No ground truth CSV reference data successfully parsed.")
#     return csv_mean, csv_std
def get_expert():
    file_pattern = './experts2/*.npz'
    file_list = sorted(glob.glob(file_pattern))
    #file_list = ['./experts2/Expert_1_actions.npz', './experts2/Expert_2_actions.npz']
    #print(file_list)
    all_normalized_csvs = []  # Added initialization back

    for file in file_list:
        try:
            with np.load(file, allow_pickle=True) as data:
                # npz archives save arrays as dictionary keys.
                # If you saved it as 'acts' inside your data pipeline, we pull it here:
                if 'acts' in data:
                    actions_matrix = data['acts']
                else:
                    # Fallback to the first available key if named differently
                    first_key = data.files[0]
                    actions_matrix = data[first_key]
            
            # The resultant force is stored in the last column (index 6)
            # Layout: [dx, dy, dz, drotx, droty, drotz, force_resultant]
            resultant = actions_matrix[:, 6]
            
            suffix = os.path.splitext(os.path.basename(file))[0]
            original_indices = np.linspace(0, 100, len(resultant))
            res_normalized = np.interp(norm_time, original_indices, resultant)
            
            all_normalized_csvs.append(pd.Series(res_normalized, name=f'Expert_{suffix}'))
            
        except Exception as e:
            print(f"Error processing local NPZ file {file}: {e}")

    if all_normalized_csvs:
        csv_df = pd.concat(all_normalized_csvs, axis=1)
        csv_mean = csv_df.mean(axis=1).values
        csv_std = csv_df.std(axis=1).values
        
        # Calculate Reference Impulse (area under curve)
        impulse_csv = np.trapezoid(csv_mean, norm_time)
        print(f'Impulse for Experimental Reference: {impulse_csv:.2f}')
    else:
        csv_mean = np.zeros_like(norm_time)
        csv_std = np.zeros_like(norm_time)
        print("Error: No ground truth NPZ reference data successfully parsed.")
    print(f"Mean Force Trajectory: {csv_mean}")
    return csv_mean, csv_std

def run_simulation(youngs_modulus, vtk_file):
    # Run the simulation for each expert trajectory
    normalized_forces = []
    for expert in ["Expert_1_actions", "Expert_2_actions", "Expert_3_actions", "Expert_4_actions", "Expert_5_actions"]:
        step_force = multiple_envs(None,
                    threshold_pos=0.0001,
                    threshold_ori=0.5,
                    maxforce=500,
                    softtissue=None,
                    youngs_modulus=youngs_modulus,
                    vtk_file=vtk_file,
                    expert=expert,log=0)
        #print(step_force)
        normalized_force = normalize_force_trajectory(step_force)
        normalized_forces.append(pd.Series(normalized_force, name=expert))

    if normalized_forces:
        force_df = pd.concat(normalized_forces, axis=1)
        force_mean = force_df.mean(axis=1).values
        force_std = force_df.std(axis=1).values
    else:
        force_df = pd.DataFrame()
        force_mean = np.array([])
        force_std = np.array([])
    print(f"Mean Force Trajectory for Young's Modulus {youngs_modulus}: {force_mean}")
    return force_df, force_mean, force_std
    
def objective_function(tuning_param,log=1):
    """Calculates the error between simulation and experiment."""
     
    # _,force_mean,_ = run_simulation(tuning_param,vtk_file)
    # #print(f"Young's Modulus: {tuning_param:.2e}, Force Mean: {force_mean}")
    # # Calculate RMSE
    # rmse = np.sqrt(np.mean((exp_forces - force_mean) ** 2))
    # print(f"Young's Modulus: {tuning_param}, RMSE: {rmse}")
    # wandb.log({"Young's Modulus": tuning_param, "RMSE": rmse})

    # _, force_mean, _ = run_simulation(tuning_param, vtk_file)
    
    # sim_impulse = np.trapezoid(force_mean, norm_time)
    # exp_impulse = np.trapezoid(exp_forces, norm_time)
    
    # # The objective is to bring the absolute energy difference down to zero
    # energy_error = np.abs(sim_impulse - exp_impulse)
    # wandb.log({"Young's Modulus": tuning_param, "energy_error": energy_error})
    # return energy_error
    """Calculates the normalized percentage error between simulation and experiment."""
    # Ensure tuning_param is a clean float scalar
    # E_value = float(tuning_param[0]) if isinstance(tuning_param, (list, np.ndarray)) else float(tuning_param)
     
    # _, force_mean, _ = run_simulation(E_value, vtk_file)
    
    # # 1. Calculate raw localized squared errors
    # squared_errors = (exp_forces - force_mean) ** 2
    # raw_rmse = np.sqrt(np.mean(squared_errors))
    
    # # 2. Extract the dynamic range of your surgeon's data (Peak-to-Peak)
    # exp_range = np.max(exp_forces) - np.min(exp_forces)
    
    # # Avoid zero division safety catch
    # if exp_range == 0:
    #     exp_range = 1e-6
        
    # # 3. Normalize the error to turn it into a relative percentage
    # nrmse = raw_rmse / exp_range
    
    # # Print with high decimal precision to track micro-movements
    # print(f"Young's Modulus: {E_value:.2e} | Raw RMSE: {raw_rmse:.5f}N | Normalized Error: {nrmse * 100:.3f}%")
    E_value = float(tuning_param[0]) if isinstance(tuning_param, (list, np.ndarray)) else float(tuning_param)
    _, force_mean, _ = run_simulation(E_value, vtk_file)
    
    # Sort arrays and grab the top 5 highest force values (approx 5% of 101 steps)
    sim_top_peaks = np.sort(force_mean)[-5:]
    exp_top_peaks = np.sort(exp_forces)[-5:]
    
    # Calculate Mean Absolute Error strictly for the highest loading states
    peak_mae = np.mean(np.abs(exp_top_peaks - sim_top_peaks))
    ## Percentage error relative to the peak of the experimental data
    peak_percentage_error = (peak_mae / np.max(exp_forces)) * 100
    if log == 1:
        wandb.log({"Young's Modulus": E_value, "Peak-Zone MAE": peak_mae})
    print(f"E: {E_value:.2e} | Peak-Zone MAE: {peak_mae:.4f}N | Peak-Zone % Error: {peak_percentage_error:.3f}%")
    return peak_percentage_error
    
    
    #return rmse


if __name__ == "__main__":
    repo_paths = ["/users/cop21cma/FracSoftGym", "/home/catherine/FractureGym",'/home/catherineabraham/FractureSoftGym/']
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
    vtk_file = "rect0009.vtk"  # Example VTK file
    exp_forces, exp_std = get_expert()
    initial_guess = 1e6
    bounds = [(1e2, 1e10)]  # Example bounds for Young's modulus
    if log == 1:
        wandb.init(project="mesh_optimisation", name="Youngs_Modulus_Optimisation",notes=commit,save_code=True)
    result = opt.minimize(objective_function, initial_guess, bounds=bounds, method='Nelder-Mead')
    if result.success:
        optimal_youngs_modulus = result.x[0]
        print(f"Optimal Young's Modulus: {optimal_youngs_modulus:.2e}")
    else: 
        print("Optimization failed:", result.message)