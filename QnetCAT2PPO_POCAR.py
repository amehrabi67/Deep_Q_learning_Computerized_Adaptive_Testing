
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### 3PL Model Functions ###
def RESPOND(item_para, theta, D=1):
    # Ensure item_para is 2D
    if item_para.ndim == 1:
        item_para = item_para.reshape(1, -1)  # Convert 1D to 2D with one row
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    # Ensure theta is broadcastable
    theta = np.atleast_1d(theta)
    p = (1 - c) / (1 + np.exp(-D * a * (theta[:, np.newaxis] - b).T)) + c
    resp = (np.random.rand(*p.shape) <= p).astype(int)
    return resp.T[0] if theta.size == 1 else resp.T  # Return 1D if single theta, else 2D

def FI(item_para, theta, D=1):
    if item_para.ndim == 1:
        item_para = item_para.reshape(1, -1)
    a = item_para[:, 0]
    b = item_para[:, 1]
    c = item_para[:, 2]
    theta = np.atleast_1d(theta)
    info = D**2 * a**2 * (1 - c) / (c + np.exp(D * a * (theta[:, np.newaxis] - b).T)) / \
           (1 + np.exp(-D * a * (theta[:, np.newaxis] - b).T))**2
    return info.T[0] if theta.size == 1 else info.T

def MLE(item_paras, resp, D=1):
    a = item_paras[:, 0]
    b = item_paras[:, 1]
    c = item_paras[:, 2]
    def mins_likelihood(x):
        logl = 0
        for i in range(len(resp)):
            p = (1 - c[i]) / (1 + np.exp(-D * a[i] * (x - b[i]))) + c[i]
            logl -= resp[i] * np.log(p) + (1 - resp[i]) * np.log(1 - p)
        return logl
    theta = minimize_scalar(mins_likelihood, bounds=(-4, 4), method='Bounded').x
    return np.array(theta).reshape(1,)

def MLE_TEST(item_paras, resp, D=1):
    theta = np.zeros(resp.shape[1])
    for j in range(resp.shape[1]):
        def mins_likelihood(x):
            logl = 0
            for i in range(resp.shape[0]):
                p = (1 - item_paras[i, 2]) / (1 + np.exp(-D * item_paras[i, 0] * (x - item_paras[i, 1]))) + item_paras[i, 2]
                logl -= resp[i, j] * np.log(p) + (1 - resp[i, j]) * np.log(1 - p)
            return logl
        theta[j] = minimize_scalar(mins_likelihood, bounds=(-4, 4), method='Bounded').x
    return np.expand_dims(theta, axis=0)

### Theta SE Calculation ###
def item_information(theta, a, b, c, D=1):
    P = c + (1 - c) / (1 + np.exp(-D * a * (theta - b)))
    Q = 1 - P
    return D**2 * a**2 * (1 - c) / (P * Q)

def test_information(theta, item_params):
    return np.sum([item_information(theta, a, b, c) for a, b, c in item_params])

def theta_se(theta, administered_items):
    info = test_information(theta, administered_items)
    return 1 / np.sqrt(info) if info > 0 else np.inf

### Policy Network Definition ###
class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, action_space):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits

### Training Function with PPO and POCAR ###
def TRAIN_PPO_POCAR(gamma, prior, memory_capacity=1000, batch_size=128, 
                    epochs=10, learning_rate=1e-3, training_size=1000, 
                    validation_size=200, validation_interval=50, epsilon_clip=0.2,
                    beta_0=1.0, beta_1=0.5, beta_2=0.5, omega=0.3):
    global best_valid
    best_valid = None

    policy_net.train()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    value_net = torch.nn.Sequential(
        nn.Linear(input_size, 50),
        nn.ReLU(),
        nn.Linear(50, 30),
        nn.ReLU(),
        nn.Linear(30, 1)
    ).to(device)
    value_optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)

    if prior == "normal":
        training_theta = np.random.randn(training_size)
    elif prior == "uniform":
        training_theta = np.random.uniform(-3, 3, training_size)

    for j in range(training_size):
        state = np.zeros(1)  # Initial theta = 0
        item_id = [np.argmin(np.abs(item_bank[:, 1]))]  # Moderate b first item
        resp = [RESPOND(item_bank[item_id[0]], training_theta[j])[0]]
        memory = []

        for i in range(test_length):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = policy_net(state_tensor)
            logits[0, item_id] = -float('inf')  # Mask used items
            action_dist = dist.Categorical(logits=logits)
            action = action_dist.sample().item()
            item_id.append(action)
            resp.append(RESPOND(item_bank[action], training_theta[j])[0])
            reward = FI(item_bank[action], training_theta[j])[0]

            # Theta update with PPO/POCAR influence
            if i == 0:
                b_value = item_bank[action, 1]
                next_theta = b_value + 1.5 if resp[-1] == 1 else b_value - 1.5  # Dramatic jump
            else:
                next_theta = MLE(item_bank[item_id], np.array(resp))[0]

            # Fairness metric: SE variance (example)
            se = theta_se(next_theta, item_bank[item_id])
            delta_t = se  # Simplified; could be variance across students in batch
            next_state = np.array([next_theta])
            memory.append([state, action, reward, next_state, delta_t])
            state = next_state

            if len(memory) >= batch_size:
                batch = memory[-batch_size:]
                states = torch.FloatTensor([m[0] for m in batch]).to(device)
                actions = torch.LongTensor([m[1] for m in batch]).to(device)
                rewards = torch.FloatTensor([m[2] for m in batch]).to(device)
                next_states = torch.FloatTensor([m[3] for m in batch]).to(device)
                deltas = torch.FloatTensor([m[4] for m in batch]).to(device)

                # Compute advantages with POCAR regularization
                values = value_net(states).squeeze()
                next_values = value_net(next_states).squeeze().detach()
                advantages = rewards + gamma * next_values - values
                advantages_pocar = beta_0 * advantages + \
                                   beta_1 * torch.min(torch.zeros_like(deltas), -deltas + omega) + \
                                   beta_2 * torch.where(deltas > omega, 
                                                        torch.min(torch.zeros_like(deltas), deltas - deltas.roll(-1, 0)), 
                                                        torch.zeros_like(deltas))

                # PPO update
                old_logits = policy_net(states).detach()
                old_dist = dist.Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(actions)
                for _ in range(epochs):
                    logits = policy_net(states)
                    dist_new = dist.Categorical(logits=logits)
                    log_probs = dist_new.log_prob(actions)
                    ratios = torch.exp(log_probs - old_log_probs)
                    surr1 = ratios * advantages_pocar
                    surr2 = torch.clamp(ratios, 1 - epsilon_clip, 1 + epsilon_clip) * advantages_pocar
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(value_net(states).squeeze(), rewards + gamma * next_values)
                    
                    optimizer.zero_grad()
                    policy_loss.backward()
                    optimizer.step()
                    
                    value_optimizer.zero_grad()
                    value_loss.backward()
                    value_optimizer.step()

        # Validation (simplified)
        if (j + 1) % validation_interval == 0:
            valid_theta = np.random.choice(training_theta, validation_size)
            state = np.zeros((1, validation_size))
            item_id = np.full((1, validation_size), np.argmin(np.abs(item_bank[:, 1])))
            resp = RESPOND(item_bank[item_id[0]], valid_theta)[np.newaxis, :]
            valid_bias = []

            for i in range(test_length):
                action = policy_net(torch.FloatTensor(state.T).to(device)).argmax(dim=1).cpu().numpy()
                item_id = np.concatenate((item_id, action[np.newaxis, :]))
                resp = np.concatenate((resp, RESPOND(item_bank[action], valid_theta)[np.newaxis, :]))
                theta_0 = np.where(i == 0, 
                                   item_bank[action, 1] + (1.5 if resp[-1] == 1 else -1.5),
                                   MLE_TEST(item_bank[item_id], resp)[0])
                state = theta_0[np.newaxis, :]
                valid_bias.append(theta_0 - valid_theta)

            bias_mean = np.mean(valid_bias, axis=1)
            print(f"Subject {j+1}, Bias Mean: {np.mean(bias_mean):.3f}")

### Testing Function ###
def TEST_PPO_POCAR(theta_test, testing_size=5000):
    policy_net.eval()
    state = np.zeros((1, testing_size))  # Initial theta = 0
    moderate_item_idx = np.argmin(np.abs(item_bank[:, 1]))
    item_id = np.full((1, testing_size), moderate_item_idx)
    resp_history = []
    theta_history = [state[0]]
    se_history = [np.full(testing_size, np.inf)]

    for i in range(test_length):
        if i > 0:  # First item is pre-selected
            state_tensor = torch.FloatTensor(state).swapaxes(0, 1).to(device)
            logits = policy_net(state_tensor)
            logits[range(testing_size), item_id.T] = -float('inf')
            action = dist.Categorical(logits=logits).sample().cpu().numpy()
            item_id = np.concatenate((item_id, action[np.newaxis, :]))

        resp = RESPOND(item_bank[item_id[-1]], theta_test)[np.newaxis, :]
        resp_history.append(resp)

        # Theta update with dramatic jump after first response
        theta_0 = np.zeros(testing_size)
        if i == 0:
            b_values = item_bank[item_id[-1], 1]
            theta_0 = np.where(resp[0] == 1, b_values + 1.5, b_values - 1.5)
        else:
            theta_0 = MLE_TEST(item_bank[item_id], np.concatenate(resp_history, axis=0))[0]
        
        state = theta_0[np.newaxis, :]
        theta_history.append(theta_0)

        # SE calculation
        se = np.array([theta_se(theta_0[j], item_bank[item_id[:, j]]) for j in range(testing_size)])
        se_history.append(se)

        print(f"Step {i+1}, Mean Theta: {np.mean(theta_0):.3f}, Mean SE: {np.mean(se):.3f}")

    # Save results
    user_id = np.repeat(np.arange(1, testing_size + 1), test_length).reshape(-1, 1)
    step = np.tile(np.arange(1, test_length + 1), testing_size).reshape(-1, 1)
    item_id_out = (item_id.T + 1).reshape(-1, 1)
    resp_out = np.concatenate(resp_history, axis=0).T.reshape(-1, 1)
    theta_est = np.array(theta_history[1:]).T.reshape(-1, 1)
    bias = (theta_est - np.repeat(theta_test, test_length)).reshape(-1, 1)
    se_out = np.array(se_history[1:]).T.reshape(-1, 1)
    data = np.hstack([user_id, step, item_id_out, resp_out, theta_est, bias, se_out])
    df = pd.DataFrame(data, columns=['userID', 'step', 'itemID', 'resp', 'theta_est', 'bias', 'theta_se'])
    df.to_csv(f'/content/drive/MyDrive/CAT_agent/records_{bank_type}_{bank_id}_PPO_POCAR_{prior}_gamma_{gamma}.csv', index=False)

### Hyperparameters and Setup ###
test_length = 40
input_size = 1
action_space = 500
bank_type = 'uncor'
bank_id = 1
prior = "uniform"
gamma = 0.1

item_bank = np.array(pd.read_csv('/content/drive/MyDrive/CAT_agent/item_bank.csv')[['a', 'b', 'c']])
theta_test = np.array(pd.read_csv('/content/drive/MyDrive/CAT_agent/test_theta.csv')['theta'])

policy_net = PolicyNet(input_size, 50, 30, action_space).to(device)

### Start Training and Testing ###
TRAIN_PPO_POCAR(gamma, prior)
TEST_PPO_POCAR(theta_test)
