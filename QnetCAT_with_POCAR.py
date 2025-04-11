# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from time import time
from torch.utils.tensorboard import SummaryWriter

# --- Setup TensorBoard Writer ---
log_dir = "./logs/CAT_DQN"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# --- Config ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_ITEMS = 50
N_STEPS = 20
N_STUDENTS = 1000
TARGET_PER_ITEM = int(N_STUDENTS * N_STEPS / N_ITEMS)
GAMMA = 0.9
LEARNING_RATE = 1e-3

# --- IRT Functions ---
def respond_3pl(item_params, theta, D=1):
    a, b, c = item_params[:, 0], item_params[:, 1], item_params[:, 2]
    p = (1 - c) / (1 + np.exp(-D * a * (theta - b))) + c
    return (np.random.rand(len(p)) < p).astype(int), p

def mle_estimation(item_params, responses, D=1):
    a, b, c = item_params[:, 0], item_params[:, 1], item_params[:, 2]
    def neg_log_likelihood(x):
        p = (1 - c) / (1 + np.exp(-D * a * (x - b))) + c
        p = np.clip(p, 1e-5, 1 - 1e-5)
        return -np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
    result = minimize_scalar(neg_log_likelihood, bounds=(-4, 4), method='bounded')
    return result.x

def FI(item_param, theta, D=1):
    a, b, c = item_param[:, 0], item_param[:, 1], item_param[:, 2]
    L = D * a * (theta - b)
    p = (1 - c) / (1 + np.exp(-L)) + c
    return D**2 * a**2 * (1 - p) * ((p - c) / (1 - c))**2 / (p * (1 - p))

# --- DQN ---
class QNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- Fairness-Aware Advantage ---
def compute_advantage(reward, delta_t, delta_tp1, beta0=1.0, beta1=0.1, beta2=0.1, omega=2.0):
    penalty1 = beta1 * np.minimum(0, -delta_t + omega)
    penalty2 = beta2 * np.minimum(0, delta_t - delta_tp1) if delta_t > omega else 0
    return beta0 * reward + penalty1 + penalty2

# --- Simulation Setup ---
def simulate_item_bank(n):
    a = np.random.uniform(0.5, 2.5, n)
    b = np.random.normal(0, 1, n)
    c = np.random.uniform(0.1, 0.3, n)
    return np.vstack([a, b, c]).T

def simulate_students(n):
    return np.random.normal(0, 1, n)

# --- Training ---
def train_agent():
    item_bank = simulate_item_bank(N_ITEMS)
    thetas = simulate_students(N_STUDENTS)
    item_counts = np.zeros(N_ITEMS)

    qnet = QNet(1, 64, N_ITEMS).to(DEVICE)
    optimizer = optim.Adam(qnet.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    total_steps = 0
    start_time = time()
    running_loss = 0.0
    running_reward = 0.0

    for student in range(N_STUDENTS):
        theta = thetas[student]
        state = torch.tensor([[theta]], dtype=torch.float32).to(DEVICE)
        used_items = []
        responses = []

        for step in range(N_STEPS):
            total_steps += 1
            q_values = qnet(state)
            for idx in used_items:
                q_values[0, idx] = -float('inf')
            action = torch.argmax(q_values).item()

            item_param = item_bank[action:action+1]
            response, _ = respond_3pl(item_param, theta)
            reward = float(FI(item_param, theta))

            used_items.append(action)
            responses.append(response[0])

            theta_hat = mle_estimation(item_bank[used_items], np.array(responses))
            next_state = torch.tensor([[theta_hat]], dtype=torch.float32).to(DEVICE)

            delta_t = abs(item_counts[action] - TARGET_PER_ITEM)
            item_counts[action] += 1
            delta_tp1 = abs(item_counts[action] - TARGET_PER_ITEM)

            adv = compute_advantage(reward, delta_t, delta_tp1)

            with torch.no_grad():
                target = adv + GAMMA * torch.max(qnet(next_state)).item()

            pred = qnet(state)[0, action]
            loss = loss_fn(pred, torch.tensor(target, dtype=torch.float32).to(DEVICE))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            writer.add_scalar("train/loss", loss.item(), total_steps)
            writer.add_scalar("train/reward", reward, total_steps)
            writer.add_scalar("train/advantage", adv, total_steps)
            writer.add_scalar("train/item_usage_variance", np.var(item_counts), total_steps)

            state = next_state
            running_loss += loss.item()
            running_reward += reward

        if (student + 1) % 100 == 0:
            elapsed = int(time() - start_time)
            mean_loss = running_loss / 100
            mean_reward = running_reward / 100

            print(f"-----------------------------------------")
            print(f"| Student {student+1}/{N_STUDENTS} | Time Elapsed: {elapsed}s")
            print(f"| Total Steps: {total_steps}")
            print(f"| Avg Loss: {mean_loss:.4f} | Avg Reward: {mean_reward:.4f}")
            print(f"-----------------------------------------")

            writer.add_scalar("train/avg_loss", mean_loss, student+1)
            writer.add_scalar("train/avg_reward", mean_reward, student+1)
            writer.add_scalar("time/fps", int(total_steps / elapsed), student+1)
            writer.add_scalar("time/elapsed", elapsed, student+1)

            running_loss = 0.0
            running_reward = 0.0

    print("Training complete.")
    print("Final Item Usage:", item_counts)
    writer.flush()

if __name__ == "__main__":
    train_agent()

