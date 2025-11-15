"""
implement_applications.py

Implements the three "Applications" described in the notebooks:
- Application 1: analytic and autograd gradients for l = (exp(wx+b)-y*)^2
- Application 2: binary classification on moons dataset (make_moons)
- Application 3: multiclass classification on synthetic spiral dataset

Saves decision-boundary plots to the working directory.
"""

import math
import os
import random

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Ensure reproducibility
random.seed(22)
np.random.seed(22)
torch.manual_seed(22)

DEVICE = torch.device("cpu")

###############################################################################
# Application 1: analytic gradients vs autograd
###############################################################################

def application_1():
    print("\n=== Application 1: analytic gradients vs autograd ===")

    def analytic_grads(w, b, x, y):
        # l = (exp(w*x + b) - y)^2
        exp_val = math.exp(w * x + b)
        diff = exp_val - y
        d_dw = 2.0 * diff * (x * exp_val)
        d_db = 2.0 * diff * exp_val
        return d_dw, d_db

    # numeric example
    w0 = 2.0
    b0 = 0.5
    x0 = 1.0
    y0 = 3.0

    dwd_analytic, dbd_analytic = analytic_grads(w0, b0, x0, y0)
    print(f"Analytic gradients: dw={dwd_analytic:.6f}, db={dbd_analytic:.6f}")

    # autograd
    w_t = torch.tensor(w0, requires_grad=True)
    b_t = torch.tensor(b0, requires_grad=True)
    x_t = torch.tensor(x0)
    y_t = torch.tensor(y0)

    loss = (torch.exp(w_t * x_t + b_t) - y_t) ** 2
    loss.backward()

    print(f"Autograd gradients: dw={w_t.grad.item():.6f}, db={b_t.grad.item():.6f}")

    # small numeric check
    eps = 1e-6
    w_t.grad.zero_(); b_t.grad.zero_()
    # finite differences for dw
    w_plus = math.exp((w0 + eps) * x0 + b0)
    loss_plus = (w_plus - y0) ** 2
    w_minus = math.exp((w0 - eps) * x0 + b0)
    loss_minus = (w_minus - y0) ** 2
    fd_dw = (loss_plus - loss_minus) / (2 * eps)

    b_plus = math.exp(w0 * x0 + (b0 + eps))
    loss_plus_b = (b_plus - y0) ** 2
    b_minus = math.exp(w0 * x0 + (b0 - eps))
    loss_minus_b = (b_minus - y0) ** 2
    fd_db = (loss_plus_b - loss_minus_b) / (2 * eps)

    print(f"Finite diff approx: dw={fd_dw:.6f}, db={fd_db:.6f}")

###############################################################################
# Application 2: Make moons binary classification
###############################################################################

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, in_features=2, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(dim=1)


def plot_decision_boundary(model, X, filename, device=DEVICE, cmap="RdYlBu"):
    model.to(device)
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.inference_mode():
        inputs = torch.tensor(grid, dtype=torch.float32).to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits).cpu().numpy()
    Z = probs.reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=50, cmap=cmap, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c='k', s=6)
    plt.title("Decision boundary")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def application_2(save_plots=True):
    print("\n=== Application 2: Moons binary classification ===")
    X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    model = SimpleBinaryClassifier(in_features=2, hidden=32).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 200
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(X_train_t.to(DEVICE))
        loss = loss_fn(logits, y_train_t.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                test_logits = model(X_test_t.to(DEVICE))
                preds = (torch.sigmoid(test_logits) >= 0.5).long()
                acc = (preds.cpu().numpy() == y_test).mean() * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | loss={loss.item():.4f} | test_acc={acc:5.2f}%")

    if save_plots:
        out = "app2_moons_decision_boundary.png"
        plot_decision_boundary(model, X, out)
        print(f"Decision boundary saved to {out}")

###############################################################################
# Application 3: spirals multiclass classification
###############################################################################

def make_spiral(n_points_per_class=100, noise=0.2, n_classes=3):
    X = []
    y = []
    for class_number in range(n_classes):
        r = np.linspace(0.0, 1, n_points_per_class)
        t = np.linspace(class_number * 4.0, (class_number + 1) * 4.0, n_points_per_class) + np.random.randn(n_points_per_class) * noise
        x1 = r * np.sin(t * 2.5)
        x2 = r * np.cos(t * 2.5)
        X.append(np.vstack([x1, x2]).T)
        y.append(np.ones(n_points_per_class, dtype=int) * class_number)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    # shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

class SimpleMultiClassifier(nn.Module):
    def __init__(self, in_features=2, hidden=64, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def plot_decision_boundary_multiclass(model, X, filename, device=DEVICE, cmap="viridis"):
    model.to(device)
    model.eval()
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    with torch.inference_mode():
        inputs = torch.tensor(grid, dtype=torch.float32).to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1).cpu().numpy()
    Z = preds.reshape(xx.shape)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=np.arange(preds.max() + 2) - 0.5, cmap=cmap, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c='k', s=6)
    plt.title("Decision boundary (multiclass)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def application_3(save_plots=True):
    print("\n=== Application 3: Spiral multiclass classification ===")
    X, y = make_spiral(n_points_per_class=200, noise=0.2, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = SimpleMultiClassifier(in_features=2, hidden=64, n_classes=3).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 400
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(X_train_t.to(DEVICE))
        loss = loss_fn(logits, y_train_t.to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            model.eval()
            with torch.inference_mode():
                test_logits = model(X_test_t.to(DEVICE))
                preds = test_logits.argmax(dim=1)
                acc = (preds.cpu().numpy() == y_test).mean() * 100.0
            print(f"Epoch {epoch:3d}/{epochs} | loss={loss.item():.4f} | test_acc={acc:5.2f}%")

    if save_plots:
        out = "app3_spirals_decision_boundary.png"
        plot_decision_boundary_multiclass(model, X, out)
        print(f"Decision boundary saved to {out}")

###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    application_1()
    application_2(save_plots=True)
    application_3(save_plots=True)
    print("\nAll applications executed. Plots (if any) saved in the working directory.")
