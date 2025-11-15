"""
run_exercises.py

Implements exercises 1-5 from `01_le_workflow_de_Pytorch_revision.ipynb`:
1) Create a linear dataset (weight=0.3, bias=0.9, >=100 points) and split 80/20
2) Create a model subclassing nn.Module using nn.Parameter (weights & bias)
3) Create loss (nn.L1Loss) and optimizer (torch.optim.SGD with lr=0.01) and train for 300 epochs
   - Test every 20 epochs
4) Make predictions and visualize (prints & optional plot)
5) Save state_dict and load into fresh model; verify predictions match

This script is designed to run quickly inside the .venv. It prints progress and saves a model file to `models/`.
"""

import os
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cpu")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# 1) Create dataset
###############################################################################

def create_linear_dataset(weight=0.3, bias=0.9, n_points=200, start=0.0, end=1.0):
    X = torch.linspace(start, end, steps=n_points).unsqueeze(1)
    y = weight * X + bias
    return X, y

###############################################################################
# 2) Model using nn.Parameter
###############################################################################

class LinearParamModel(nn.Module):
    """Simple linear model using nn.Parameter for weight and bias."""
    def __init__(self):
        super().__init__()
        # initialize randomly (similar to notebook behavior)
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

###############################################################################
# Helper: train/test loops
###############################################################################

def train(model, X_train, y_train, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, X_test, y_test, loss_fn):
    model.eval()
    with torch.inference_mode():
        y_pred = model(X_test)
        loss = loss_fn(y_pred, y_test)
    return loss.item(), y_pred

###############################################################################
# Main: put exercises together
###############################################################################

def run_exercises():
    print("\nRunning exercises 1-5 from notebook (linear regression workflow)")

    # 1) Create dataset
    weight_true = 0.3
    bias_true = 0.9
    n_points_total = 200  # >= 100
    X, y = create_linear_dataset(weight=weight_true, bias=bias_true, n_points=n_points_total)

    # Split 80/20
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Dataset created: total={len(X)}, train={len(X_train)}, test={len(X_test)}")

    # 2) Model
    model = LinearParamModel().to(DEVICE)
    print("Initial model.state_dict():")
    print(model.state_dict())

    # 3) Loss and optimizer
    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)

    epochs = 300
    train_losses = []
    test_losses = []
    epochs_record = []

    for epoch in range(1, epochs + 1):
        tr_loss = train(model, X_train.to(DEVICE), y_train.to(DEVICE), loss_fn, optimizer)
        if epoch % 20 == 0 or epoch == 1:
            te_loss, _ = test(model, X_test.to(DEVICE), y_test.to(DEVICE), loss_fn)
            train_losses.append(tr_loss)
            test_losses.append(te_loss)
            epochs_record.append(epoch)
            print(f"Epoch {epoch:3d}/{epochs} | Train MAE: {tr_loss:.6f} | Test MAE: {te_loss:.6f}")

    # 4) Make predictions on test set and plot
    _, y_pred_test = test(model, X_test.to(DEVICE), y_test.to(DEVICE), loss_fn)
    y_pred_test = y_pred_test.detach().cpu()

    # Print a few examples
    print("\nSome test predictions (first 6):")
    for i in range(min(6, len(X_test))):
        print(f"x={X_test[i].item():.3f} | y_true={y_test[i].item():.3f} | y_pred={y_pred_test[i].item():.3f}")

    # Plot predictions vs true
    plt.figure(figsize=(6,4))
    plt.scatter(X_train.numpy(), y_train.numpy(), label='train', s=8)
    plt.scatter(X_test.numpy(), y_test.numpy(), label='test', s=8)
    plt.scatter(X_test.numpy(), y_pred_test.numpy(), label='pred', s=10, c='r')
    plt.legend()
    plt.title('Linear model: true vs predicted (test)')
    plt.tight_layout()
    plot_path = Path('ex1_predictions.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Prediction plot saved to {plot_path}")

    # 5) Save and load state_dict
    save_path = MODEL_DIR / "exercise_linear_state_dict.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model state_dict to {save_path}")

    # Create a fresh instance and load
    loaded_model = LinearParamModel().to(DEVICE)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()

    # Verify predictions match
    with torch.inference_mode():
        orig_preds = model(X_test.to(DEVICE)).cpu()
        loaded_preds = loaded_model(X_test.to(DEVICE)).cpu()

    same = torch.allclose(orig_preds, loaded_preds, atol=1e-6)
    print(f"Loaded model predictions match original: {same}")

    if not same:
        # show sample differences
        diffs = (orig_preds - loaded_preds).abs()
        print("Max difference:", diffs.max().item())

    # Save training curve
    plt.figure()
    plt.plot(epochs_record, train_losses, label='train_mae')
    plt.plot(epochs_record, test_losses, label='test_mae')
    plt.xlabel('epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Loss curve (sampled every 20 epochs)')
    plt.tight_layout()
    plt.savefig('ex1_loss_curve.png')
    plt.close()
    print("Saved loss curve to ex1_loss_curve.png")

    print('\nExercises 1-5 completed.')

if __name__ == '__main__':
    run_exercises()
