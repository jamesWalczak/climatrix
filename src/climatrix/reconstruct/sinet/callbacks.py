from typing import Callable

import torch


class EarlyStopping:
    step: Callable[[float, torch.nn.Module | None], bool]

    def __init__(
        self,
        patience: int | None,
        delta=0.0,
        checkpoint_path=None,
        verbose=False,
    ):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_metric = float("inf")

        if self.patience is None:
            self.step = self.dummy_call
        else:
            self.step = self.call

    def dummy_call(self, val_metric, model=None):
        return False

    def call(self, val_metric, model=None):
        score = -val_metric

        if self.best_score is None:
            self.best_score = score
            if self.checkpoint_path and model:
                self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f"[EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.checkpoint_path and model:
                self.save_checkpoint(val_metric, model)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_metric, model):
        """Save the model state_dict to the checkpoint path."""
        if self.verbose:
            print(
                f"[EarlyStopping] Validation metric improved "
                f"({self.best_val_metric:.6f} -> {val_metric:.6f}). Saving model."
            )
        torch.save(model.state_dict(), self.checkpoint_path)
        self.best_val_metric = val_metric
