import math
import json
import random
import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from scipy.stats import norm

def irt_probability(theta, a, b, c):
    return c + (1 - c) / (1 + math.exp(-1.7 * a * (theta - b)))

def eap_estimate(responses, a_vals, b_vals, c_vals, nqt=31, prior_mean=0, prior_sd=1):
    thetas = np.linspace(-4, 4, nqt)
    prior = norm.pdf(thetas, loc=prior_mean, scale=prior_sd)
    likelihood = np.ones(nqt)
    for (r, a, b, c) in zip(responses, a_vals, b_vals, c_vals):
        p_grid = c + (1 - c) / (1 + np.exp(-1.7 * a * (thetas - b)))
        likelihood *= (p_grid**r) * ((1-p_grid)**(1-r))
    posterior = likelihood * prior
    posterior /= (posterior.sum() + 1e-12)
    return np.sum(thetas * posterior)

def fisher_information(a, b, c, theta):
    D2 = 2.89
    num = D2 * (a ** 2) * (1 - c)
    denom1 = c + np.exp(1.7 * a * (theta - b))
    denom2 = (1 + np.exp(-1.7 * a * (theta - b))) ** 2
    return num / (denom1 * denom2)

def standard_error(a_vals, b_vals, c_vals, theta):
    total_info = sum(fisher_information(a, b, c, theta) for a, b, c in zip(a_vals, b_vals, c_vals))
    return 1.0 / np.sqrt(total_info + 1e-12)

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
            "item_id": i+1,
            "difficulty": round(float(difficulty_values[i]), 3),
            "discrimination": round(float(discrimination_values[i]), 3),
            "guessing": round(float(guessing_values[i]), 3),
            "slip": round(float(slip_values[i]), 3),
            "learning_objectives": json.dumps(lo_dict)
        })
    return pd.DataFrame(item_bank)

class CATEnv(Env):
    """
    A Gymnasium-style environment for a simple CAT.
    - Action: Discrete(num_items) -> select an item index (0-indexed).
    - Observation: current estimated ability (theta) as a single float.
    - Episode terminates when at least min_items are administered and either:
         SE(theta) <= 0.3 or maximum items reached.
    - Reward: change in theta plus bonus if the response is correct.
    - Also records administered item details (item parameters and response).
    """
    def __init__(self, item_bank_df: pd.DataFrame, agent_true_ability: float = 0.0,
                 min_items: int = 10, max_items: int = 30):
        super().__init__()
        self.item_bank_df = item_bank_df.copy().reset_index(drop=True)
        self.num_items = len(self.item_bank_df)
        self.agent_true_ability = agent_true_ability
        self.min_items = min_items
        self.max_items = max_items
        self.action_space = spaces.Discrete(self.num_items)
        self.observation_space = spaces.Box(low=-4, high=4, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.administered_indices = []
        self.responses = []
        self.a_vals = []
        self.b_vals = []
        self.c_vals = []
        self.theta_est = 0.0
        self.steps = 0
        # New: Track administered item details (full row details and response)
        self.administered_item_details = []
        return np.array([self.theta_est], dtype=np.float32), {}

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def step(self, action):
        if action in self.administered_indices:
            valid_actions = [i for i in range(self.num_items) if i not in self.administered_indices]
            if not valid_actions:
                return np.array([self.theta_est], dtype=np.float32), 0.0, True, False, {}
            action = random.choice(valid_actions)

        row = self.item_bank_df.iloc[action]
        # Record detailed info from the item (as a dictionary) plus the eventual response:
        item_detail = row.to_dict()
        a = float(row["discrimination"])
        b = float(row["difficulty"])
        c = float(row["guessing"])

        p_correct = irt_probability(self.agent_true_ability, a, b, c)
        r = 1 if np.random.rand() < p_correct else 0

        # Save the detailed information including the response:
        item_detail.update({"response": r, "p_correct": p_correct})
        self.administered_item_details.append(item_detail)

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
        return obs, float(reward), done, False, info

    def sample_next_item(self):
        remaining = self.item_bank_df[~self.item_bank_df["item_id"].isin(
            [self.item_bank_df.iloc[i]["item_id"] for i in self.administered_indices]
        )]
        if remaining.empty:
            return None
        return int(remaining.sample(1)["item_id"])
