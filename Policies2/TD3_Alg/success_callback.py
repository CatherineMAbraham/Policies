import os
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class StopTrainingOnSuccessRate(BaseCallback):
    """
    Stops training if:
    1. Success rate reaches 1.0 (100%) immediately.
    2. Success rate reaches `success_threshold` AND fails to improve for 
       `max_no_improvement_evals` consecutive evaluations.
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
        self.eval_count = 0
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
        
        if hasattr(model, "save_replay_buffer"):
            rb_path = os.path.join(self.model_path, f"{self.model_name}-rb")
            model.save_replay_buffer(rb_path)
            
        if self.verbose >= 1:
            print(f"Model and env stats saved to {self.model_path}")
        
    def _on_step(self) -> bool:
        assert self.parent is not None, "StopTrainingOnSuccessRate must be used as a child callback inside EvalCallback"

        # IMPORTANT: Sync with parent evaluation cycle (only execute on evaluation steps)
        if self.n_calls % self.parent.eval_freq != 0:
            return True

        self.eval_count += 1
        if self.eval_count <= self.min_evals:
            return True

        # Extract success rate from evaluation buffer
        if len(self.parent._is_success_buffer) == 0:
            return True
            
        success_rate = float(np.mean(self.parent._is_success_buffer))

        # 1. Immediate Stop if 100% success rate is hit
        if success_rate >= 1.0:
            if self.verbose >= 1:
                print("100% success rate reached! Saving model and stopping training.")
            self.save_model(self.parent.model)
            wandb.summary['best_success_rate'] = 1.0
            return False

        # Mark threshold met if we hit target success rate
        if success_rate >= self.success_threshold:
            self.threshold_met = True

        # 2. Track best model and save
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.no_improvement_evals = 0
            
            if self.verbose >= 1:
                print(f"New best success rate: {self.best_success_rate:.2f}. Saving model...")
            self.save_model(self.parent.model)
            wandb.summary['best_success_rate'] = self.best_success_rate
        else:
            # 3. Only count non-improving evaluations AFTER meeting threshold
            if self.threshold_met:
                self.no_improvement_evals += 1
                if self.verbose >= 1:
                    print(f"No improvement for {self.no_improvement_evals}/{self.max_no_improvement_evals} evaluations.")

        # 4. Stop if patience limit reached
        if self.no_improvement_evals >= self.max_no_improvement_evals:
            if self.verbose >= 1:
                print(f"Stopping training: no success improvement for {self.max_no_improvement_evals} consecutive evaluations post-threshold.")
            return False

        return True