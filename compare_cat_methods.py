import math
import json
import random
import numpy as np
import pandas as pd
from scipy.stats import norm

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
# MPI Function for Traditional CAT Item Selection
# ----------------------------
def MPI(theta, administered_ids, item_bank_df):
    valid_items = item_bank_df[~item_bank_df["item_id"].isin(administered_ids)]
    if valid_items.empty:
        return None
    PI = []
    for idx, row in valid_items.iterrows():
        a = float(row["discrimination"])
        b = float(row["difficulty"])
        c = float(row["guessing"])
        PI.append(fisher_information(a, b, c, theta))
    max_index = np.argmax(PI)
    return int(valid_items.iloc[max_index]["item_id"])

# ----------------------------
# Traditional CAT Simulation Using MPI
# ----------------------------
def simulate_cat_session_MPI(item_bank_df, agent_true_ability=1.5, lk=10, uk=30):
    administered_ids = []  # Stores item_id's (1-indexed)
    responses = []
    a_vals = []
    b_vals = []
    c_vals = []

    # Initial selection: choose the first item (lowest item_id)
    first_item = item_bank_df.sort_values("item_id").iloc[0]
    a = float(first_item["discrimination"])
    b = float(first_item["difficulty"])
    c = float(first_item["guessing"])
    p_correct = irt_probability(agent_true_ability, a, b, c)
    r = 1 if np.random.rand() < p_correct else 0
    administered_ids.append(int(first_item["item_id"]))
    responses.append(r)
    a_vals.append(a)
    b_vals.append(b)
    c_vals.append(c)
    theta_est = eap_estimate(responses, a_vals, b_vals, c_vals)
    steps = 1

    while True:
        se = standard_error(a_vals, b_vals, c_vals, theta_est)
        if steps >= lk and (se <= 0.3 or steps >= uk):
            break
        next_item_id = MPI(theta_est, administered_ids, item_bank_df)
        if next_item_id is None:
            break
        row = item_bank_df[item_bank_df["item_id"] == next_item_id].iloc[0]
        a = float(row["discrimination"])
        b = float(row["difficulty"])
        c = float(row["guessing"])
        p_correct = irt_probability(agent_true_ability, a, b, c)
        r = 1 if np.random.rand() < p_correct else 0
        administered_ids.append(next_item_id)
        responses.append(r)
        a_vals.append(a)
        b_vals.append(b)
        c_vals.append(c)
        theta_est = eap_estimate(responses, a_vals, b_vals, c_vals)
        steps += 1

    se_final = standard_error(a_vals, b_vals, c_vals, theta_est)
    return theta_est, administered_ids, responses, steps, se_final

# ----------------------------
# Main: Compare Traditional CAT (MPI-based)
# ----------------------------
def main():
    item_bank_df = generate_item_bank(num_items=100, seed=123)
    true_ability = 1.5
    theta_est, administered_ids, responses, steps, se_final = simulate_cat_session_MPI(
        item_bank_df, agent_true_ability=true_ability, lk=10, uk=30
    )
    print("Traditional CAT (MPI-based) Simulation:")
    print(f"True Ability: {true_ability}")
    print(f"Estimated Theta: {theta_est:.3f}")
    print(f"Number of items administered: {steps}")
    print(f"Standard Error: {se_final:.3f}")
    print("Administered Items and Responses:")
    for item, resp in zip(administered_ids, responses):
        print(f"Item {item}: response = {resp}")

if __name__ == '__main__':
    main()
