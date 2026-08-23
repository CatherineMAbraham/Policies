import os
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

class StopTrainingOnSuccessRate(BaseCallback):
    """
    Stops training if:
    1. Success rate reaches 1.0 (100%) IMMEDIATELY.
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
        # Step handler must return True to maintain normal environment stepping
        return True

    def _on_event(self) -> bool:
        """Called by EvalCallback immediately after evaluation finishes."""
        assert self.parent is not None, "StopTrainingOnSuccessRate must be passed as callback_after_eval inside EvalCallback"

        self.eval_count += 1

        # Extract success rate from evaluation buffer
        if len(self.parent._is_success_buffer) == 0:
            return True
            
        success_rate = float(np.mean(self.parent._is_success_buffer))

        # --- IMMEDIATE HARD STOP AT 100% SUCCESS ---
        # Bypasses min_evals checks to stop training instantly on a perfect evaluation run
        if success_rate >= 1.0:
            if self.verbose >= 1:
                print("\n" + "="*50)
                print("PERFECT SCORE: 100% success rate reached!")
                print("Saving final model and stopping training IMMEDIATELY.")
                print("="*50 + "\n")
            
            self.best_success_rate = 1.0
            self.save_model(self.parent.model)
            if wandb.run is not None:
                wandb.summary['best_success_rate'] = 1.0
            
            return False  # Returning False instantly terminates model.learn()

        # Respect min_evals for non-perfect evaluation checks
        if self.eval_count <= self.min_evals:
            return True

        # Check if target success threshold met
        if success_rate >= self.success_threshold:
            self.threshold_met = True

        # Track best success rate & save model
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            self.no_improvement_evals = 0
            
            if self.verbose >= 1:
                print(f"New best success rate: {self.best_success_rate:.2f}. Saving model...")
            self.save_model(self.parent.model)
            if wandb.run is not None:
                wandb.summary['best_success_rate'] = self.best_success_rate
        else:
            # Increment patience counter post-threshold
            if self.threshold_met:
                self.no_improvement_evals += 1
                if self.verbose >= 1:
                    print(f"No improvement for {self.no_improvement_evals}/{self.max_no_improvement_evals} evaluations post-threshold.")

        # Stop training if patience limit reached
        if self.no_improvement_evals >= self.max_no_improvement_evals:
            if self.verbose >= 1:
                print(f"Stopping training: no success improvement for {self.max_no_improvement_evals} consecutive evaluations post-threshold.")
            return False

        return True