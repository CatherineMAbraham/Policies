import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import wandb

class StopTrainingOnSuccessRate(BaseCallback):
    """
    Stop training early once success rate reaches a threshold, or if there is no 
    improvement for N consecutive evaluations. Saves the model when success rate is >= 90%.
    """
    parent: EvalCallback

    def __init__(self, 
                 vec_env, 
                 max_no_improvement_evals: int, 
                 success_threshold: float,
                 min_evals: int = 0, 
                 verbose: int = 0,
                 model_name: str = "best_model",
                 model_save_path: str = None):
        super().__init__(verbose=verbose)
        self.vec_env = vec_env
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.success_threshold = success_threshold
        self.best_success_rate = -np.inf
        self.no_improvement_evals = 0
        self.model_save_path = model_save_path
        self.model_name = model_name
        self.threshold_met = False
        
        if self.model_save_path:
            self.model_path = os.path.join(self.model_save_path, self.model_name)
            os.makedirs(self.model_path, exist_ok=True)
        
    def save_model(self, model):
        if self.model_save_path is None:
            return
        model.save(os.path.join(self.model_path, self.model_name))
        stats_path = os.path.join(self.model_path, "vec_normalize.pkl")
        self.vec_env.save(stats_path)
        
        # Save replay buffer if it exists (e.g., DQN, SAC, TD3)
        # don't need this for eval so saves space 
        # if hasattr(model, "save_replay_buffer"):
        #     rb_path = os.path.join(self.model_path, f"{self.model_name}-rb.zip")
        #     model.save_replay_buffer(rb_path)
            
        if self.verbose >= 1:
            print(f"Model and env stats saved to {self.model_path}")
        
    def _on_step(self) -> bool:
        assert self.parent is not None, "StopTrainingOnSuccessRate must be used as a callback inside EvalCallback"

        continue_training = True

        # Ensure this code only evaluates during an actual evaluation step, 
        # not every single environment step (which would destroy performance).
        if self.n_calls > self.min_evals:
            success_rate = np.mean(self.parent._is_success_buffer)

            # 1. Check if we reached the ultimate stopping success threshold
            if success_rate >= self.success_threshold:
                self.threshold_met = True
                if self.verbose >= 1:
                    print(f"Success threshold ({self.success_threshold:.2f}) met with rate {success_rate:.2f}! Stopping training.")
                self.save_model(self.parent.model)
                wandb.summary['best_success_rate'] = success_rate
                if self.success_threshold ==1:
                    continue_training = False
                return False  # Stops the training immediately

            # 2. Track "new best" models and handle saving when success rate is >= 0.90
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                self.no_improvement_evals = 0
                
                
                # Only save if we have achieved a NEW best success rate that is also >= 90%
                if success_rate >= 0.0:
                    if self.verbose >= 1:
                        print(f"New best success rate: {self.best_success_rate:.2f} (>= 0.90). Saving model...")
                    self.save_model(self.parent.model)
                
                wandb.summary['best_success_rate'] = self.best_success_rate
            else:
                # 3. If no improvement, increment counter (only after we've met our first benchmark)
                if self.threshold_met:
                    self.no_improvement_evals += 1

            # 4. Handle early stopping due to patience limit
            if self.no_improvement_evals >= self.max_no_improvement_evals:
                if self.verbose >= 1:
                    print(f"Stopping training: no success improvement for {self.max_no_improvement_evals} consecutive evaluations.")
                continue_training = False

        return continue_training