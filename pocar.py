import math
import json
import random
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Fairness Metric Function
# ----------------------------
def fairness_metric(theta):
    # We desire the estimated ability to be at least 1.6.
    # Fairness violation Δ(s) = max(0, target - theta)
    target = 1.6
    return max(0, target - theta)

# ----------------------------
# IRT and EAP Helper Functions
# ----------------------------
def irt_probability(theta, a, b, c):
    return c + (1 - c) / (1 + math.exp(-1.7 * a * (theta - b)))

def eap_estimate(responses, a_vals, b_vals, c_vals, nqt=31, prior_mean=0, prior_sd=1):
    thetas = np.linspace(-4, 4, nqt)
    prior = norm.pdf(thetas, loc=prior_mean, scale=prior_sd)
    likelihood = np.ones(nqt)
    for (r, a, b, c) in zip(responses, a_vals, b_vals, c_vals):
        p_grid = c + (1 - c) / (1 + np.exp(-1.7 * a * (thetas - b)))
        likelihood *= (p_grid**r) * ((1 - p_grid)**(1 - r))
    posterior = likelihood * prior
    posterior /= (posterior.sum() + 1e-12)
    return np.sum(thetas * posterior)

def fisher_information(a, b, c, theta):
    D2 = 2.89
    num = D2 * (a**2) * (1 - c)
    denom1 = c + np.exp(1.7 * a * (theta - b))
    denom2 = (1 + np.exp(-1.7 * a * (theta - b)))**2
    return num / (denom1 * denom2)

def standard_error(a_vals, b_vals, c_vals, theta):
    total_info = sum(fisher_information(a, b, c, theta) for a, b, c in zip(a_vals, b_vals, c_vals))
    return 1.0 / np.sqrt(total_info + 1e-12)

# ----------------------------
# Generate Item Bank
# ----------------------------
def generate_item_bank(num_items=100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    difficulty_values = np.clip(np.random.normal(0, 1, num_items), -3, 3)
    discrimination_values = np.clip(np.random.lognormal(0, 0.5, num_items), 0, 3)
    guessing_values = np.random.uniform(0.05, 0.3, num_items)
    slip_values = np.random.uniform(0.01, 0.3, num_items)
    lo_keys = ["lo1", "lo2", "lo3", "lo4"]
    item_bank = []
    for i in range(num_items):
        num_lo = random.randint(1, 3)
        chosen_los = random.sample(lo_keys, num_lo)
        lo_dict = {lo: random.choice([0, 1]) for lo in chosen_los}
        item_bank.append({
            "item_id": i + 1,
            "difficulty": round(float(difficulty_values[i]), 3),
            "discrimination": round(float(discrimination_values[i]), 3),
            "guessing": round(float(guessing_values[i]), 3),
            "slip": round(float(slip_values[i]), 3),
            "learning_objectives": json.dumps(lo_dict)
        })
    return pd.DataFrame(item_bank)

# ----------------------------
# CAT Environment (simplified, Gym-style)
# ----------------------------
class CATEnv:
    def __init__(self, item_bank_df, agent_true_ability=0.0, min_items=10, max_items=30):
        self.item_bank_df = item_bank_df.copy().reset_index(drop=True)
        self.num_items = len(self.item_bank_df)
        self.agent_true_ability = agent_true_ability
        self.min_items = min_items
        self.max_items = max_items
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.administered_indices = []
        self.responses = []
        self.a_vals = []
        self.b_vals = []
        self.c_vals = []
        self.theta_est = 0.0
        self.steps = 0
        # For logging detailed info:
        self.administered_item_details = []
        return np.array([self.theta_est], dtype=np.float32)

    def step(self, action):
        if action in self.administered_indices:
            valid_actions = [i for i in range(self.num_items) if i not in self.administered_indices]
            if not valid_actions:
                return np.array([self.theta_est], dtype=np.float32), 0.0, True, False, {}
            action = random.choice(valid_actions)
        row = self.item_bank_df.iloc[action]
        detail = row.to_dict()
        a = float(row["discrimination"])
        b = float(row["difficulty"])
        c = float(row["guessing"])
        p_correct = irt_probability(self.agent_true_ability, a, b, c)
        r = 1 if np.random.rand() < p_correct else 0

        # Record detailed info with response
        detail.update({"response": r, "p_correct": p_correct})
        self.administered_item_details.append(detail)

        self.administered_indices.append(action)
        self.responses.append(r)
        self.a_vals.append(a)
        self.b_vals.append(b)
        self.c_vals.append(c)

        old_theta = self.theta_est
        self.theta_est = eap_estimate(self.responses, self.a_vals, self.b_vals, self.c_vals)
        self.steps += 1

        se = standard_error(self.a_vals, self.b_vals, self.c_vals, self.theta_est)
        done = False
        if self.steps >= self.min_items and (se <= 0.3 or self.steps >= self.max_items):
            done = True

        reward = (self.theta_est - old_theta) + (0.5 if r == 1 else 0)
        obs = np.array([self.theta_est], dtype=np.float32)
        info = {"p_correct": p_correct, "se": se, "score": r}
        # Return observation, reward, terminated, truncated, info
        return obs, float(reward), done, False, info

    def sample_action(self):
        # For a baseline, just randomly select an item index
        valid_actions = [i for i in range(self.num_items) if i not in self.administered_indices]
        if valid_actions:
            return random.choice(valid_actions)
        return None

# ----------------------------
# Neural Network Models for Policy and Value
# ----------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # Output probability distribution over actions
        )
    def forward(self, x):
        return self.net(x)
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Replay Buffer and Data Collection for POCAR
# ----------------------------
class ReplayBuffer:
    def __init__(self):
        self.buffer = []
    def add(self, transition):
        self.buffer.append(transition)
    def sample(self):
        return self.buffer
    def clear(self):
        self.buffer = []

def collect_trajectories(env, policy_net, num_episodes=10, max_steps=30, gamma=0.99):
    buffer = ReplayBuffer()
    for ep in range(num_episodes):
        state = env.reset(seed=ep)
        # Compute fairness metric for state
        delta_t = fairness_metric(state[0])
        for t in range(max_steps):
            action, log_prob = policy_net.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            delta_next = fairness_metric(next_state[0])
            buffer.add((state, action, reward, next_state, delta_t, delta_next, log_prob))
            state = next_state
            delta_t = delta_next
            if done:
                break
    return buffer

# ----------------------------
# Advantage Computation and Fairness Regularization
# ----------------------------
def compute_advantages(buffer, value_net, gamma=0.99):
    transitions = buffer.sample()
    advantages = []
    returns = []
    for (state, action, reward, next_state, delta_t, delta_next, log_prob) in transitions:
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        v = value_net(state_tensor).item()
        v_next = value_net(next_state_tensor).item()
        ret = reward + gamma * v_next
        adv = ret - v
        advantages.append(adv)
        returns.append(ret)
    return advantages, returns

def apply_fairness_regularization(advantages, transitions, beta0, beta1, beta2, omega):
    # transitions contains fairness metrics: delta_t and delta_next
    A_fair_list = []
    for adv, trans in zip(advantages, transitions):
        _, _, _, _, delta_t, delta_next, _ = trans
        A_fair = beta0 * adv + beta1 * min(0, -delta_t + omega)
        if delta_t > omega:
            A_fair += beta2 * min(0, delta_t - delta_next)
        A_fair_list.append(A_fair)
    return A_fair_list

# ----------------------------
# Policy Update (PPO-like with Clipping and Fairness Regularization)
# ----------------------------
def update_policy(buffer, policy_net, old_policy_net, value_net, optimizer_policy, optimizer_value,
                  epsilon=0.2, beta0=1.0, beta1=1.0, beta2=1.0, omega=0.5, epochs=5):
    transitions = buffer.sample()
    states = torch.FloatTensor(np.vstack([trans[0] for trans in transitions]))
    actions = torch.LongTensor([trans[1] for trans in transitions]).unsqueeze(1)
    # Detach old_log_probs so they are not part of the current computational graph.
    old_log_probs = torch.stack([trans[6] for trans in transitions]).unsqueeze(1).detach()

    advantages, returns = compute_advantages(buffer, value_net)
    advantages = torch.FloatTensor(advantages).unsqueeze(1)

    # Apply fairness regularization to the advantages.
    A_fair = apply_fairness_regularization(advantages.squeeze(1).tolist(), transitions, beta0, beta1, beta2, omega)
    A_fair = torch.FloatTensor(A_fair).unsqueeze(1)

    for epoch in range(epochs):
        optimizer_policy.zero_grad()
        optimizer_value.zero_grad()
        # Recompute the forward pass each epoch
        probs = policy_net(states)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions.squeeze(1)).unsqueeze(1)

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * A_fair
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A_fair
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        value_preds = value_net(states)
        value_loss = torch.mean((value_preds - torch.FloatTensor(returns).unsqueeze(1))**2)

        total_loss = policy_loss + 0.5 * value_loss

        total_loss.backward()
        optimizer_policy.step()
        optimizer_value.step()

    # Update old policy with new policy parameters.
    old_policy_net.load_state_dict(policy_net.state_dict())
    buffer.clear()
    return total_loss.item()


# ----------------------------
# Main Training Loop for POCAR
# ----------------------------
def train_pocar(num_iterations=1000, num_episodes=5, max_steps=30):
    # Generate item bank and create environment
    item_bank_df = generate_item_bank(num_items=100, seed=123)
    env = CATEnv(item_bank_df=item_bank_df, agent_true_ability=1.5, min_items=10, max_items=30)

    # Define networks: state is 1D, action space size equals num_items
    input_dim = 1
    output_dim = env.num_items
    policy_net = PolicyNetwork(input_dim, output_dim)
    old_policy_net = PolicyNetwork(input_dim, output_dim)
    old_policy_net.load_state_dict(policy_net.state_dict())
    value_net = ValueNetwork(input_dim)

    optimizer_policy = optim.Adam(policy_net.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    # Hyperparameters for fairness regularization
    beta0, beta1, beta2, omega = 1.0, 1.0, 1.0, 0.5
    gamma = 0.99

    for iter in range(num_iterations):
        # Collect trajectories with fairness metrics
        buffer = collect_trajectories(env, policy_net, num_episodes=num_episodes, max_steps=max_steps, gamma=gamma)
        loss = update_policy(buffer, policy_net, old_policy_net, value_net, optimizer_policy, optimizer_value,
                             epsilon=0.2, beta0=beta0, beta1=beta1, beta2=beta2, omega=omega, epochs=5)
        if iter % 100 == 0:
            print(f"Iteration {iter}: Total Loss = {loss:.4f}, Current Theta Estimate = {env.theta_est:.3f}")
    # After training, test the policy
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = policy_net.select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Test completed: Total reward = {total_reward:.3f}, Final theta = {state[0]:.3f}")
    print("Administered Items:", env.administered_indices)
    print("Administered Items Details:")
    for detail in env.administered_item_details:
        print(detail)

if __name__ == '__main__':
    train_pocar(num_iterations=200, num_episodes=5, max_steps=30)
