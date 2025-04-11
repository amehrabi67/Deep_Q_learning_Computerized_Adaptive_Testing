# ECE570-AI-chek3
In this Repository:
cat-agent-project/
├── cat_agent/              # Main package directory
│   ├── __init__.py         # Marks this as a Python package
│   ├── train_agent.py      # PPO training script
│   ├── cat_env.py          # CAT environment definition
│   ├── compare_cat_methods.py  # Traditional CAT simulation
│   ├── pocar.py            # POCAR implementation
│   ├── qnetcat1.py         # DQN with normal prior (renamed to avoid special chars)
│   ├── qnetcat2.py         # DQN with uniform prior and plotting
│   ├── qnetcat_with_pocar.py  # DQN with POCAR fairness 
│   ├── QnetCAT2.py #A better # DQN CAT environment (main code)
│   ├── QnetCAT2Huber.py  # DQN with Huber CAT  environment (main code)
│   └── QnetCAT2HuberPPO.py  # DQN with Huber CAT with PPO environment (main code)
├── requirements.txt        # Dependencies list
├── setup.py                # Installation script
├── README.md               # Project documentation
├── LICENSE                 # License file (e.g., MIT)


Explanation:
This document provides detailed explanations for each .py file in a project centered on training reinforcement learning agents for 
Computerized Adaptive Testing (CAT). Each file serves a distinct purpose, from environment setup to training different RL algorithms 
(PPO, DQN, POCAR, POCAR_DQN) and comparing their performance against traditional CAT methods. Below, each file is described with its logic, 
formulation, and how to run it.

Steps to Run the codes and analysis:
Before everything, please run the code of "setup.py". To embark on this journey, follow these carefully crafted steps with delight. 
+++++Step 1 invites you to mount the files in your chosen realm, be it the ethereal cloud of Google Colab or the grounded domain of a Conda environment. If you dance within Colab’s embrace, 
christen the destination path as /content/drive/MyDrive/CAT_agent, summoning the magic of the drive module from google.colab with drive.mount('/content/drive'), then gracefully gliding to that directory with a command to settle in /content/drive/MyDrive/CAT_agent. 
+++++Step 2 beckons you to weave the tapestry of dependencies by unfurling the requirements.txt file, casting the incantation pip install -r requirements.txt to bring forth the tools of creation. 
+++++Step 3 unveils the radiant heart of the source code, adorned with my bespoke edition of the CAT environment, a masterpiece of adaptive testing artistry. 
In step 3.1, awaken cat_env.py, my cherished creation, to lay the foundation of a world where questions bloom in harmony with ability. Then, in step 
3.2, let train_agent.py take flight, training a PPO-based agent with elegance and precision, weaving a symphony of learning within this enchanted domain. Next, in step 
3.3, summon compare_cat_methods.py to paint a canvas of comparison, where methods vie for supremacy under the watchful gaze of performance metrics. 
For step 3.4, breathe life into visualization by whispering %load_ext tensorboard to summon the extension, followed by %tensorboard --logdir=./logs/ to unveil a breathtaking gallery of PPO’s soul—loss, rewards, and entropy dancing in vivid hues; here, immerse yourself in the splendor, selecting each agent to witness its unique tale unfold in the analysis graphs. 
+++++Step 4 introduces the majestic PoCAR, a tapestry of Markov’s long-term strategies woven with threads of patience and foresight. 
In step 4.1, ignite pocar.py to behold a grand spectacle—agents traversing time, their abilities shimmering under the guiding star of fairness constraints, revealing the beauty of sustained strategy. 
+++++Step 5 heralds the celestial union of Q-networks and PoCAR, a fusion of intellect and equity. 
In step 5.1, launch QnetCAT1.py to witness a DQN agent, a knight of precision, selecting items with the grace of estimated ability. 
Then, in step 5.2, unleash QnetCAT2.py, where simulated students join the stage, their performance tracked in a ballet of data and insight. 
In step 5.3, exalt QnetCAT_with_POCAR.py, blending Q-learning’s valor with PoCAR’s fairness, crafting a harmonious crescendo of adaptive brilliance. 
In step 5.4, chant %load_ext tensorboard and swiftly follow with %tensorboard --logdir=./logs/CAT_DQN to unveil a dazzling, interactive tapestry—loss unfurling like a river, rewards rising like dawn, and metrics shimmering with fairness and variance; lose yourself in this radiant display, zooming into the agents’ journeys and savoring their evolution under the twin banners of constraint and growth.
Now we are in last step. 
+++++In step 6 you need to run QnetCAT2.py, QnetCAT2Huber.py, and QnetCAT2HuberPPO.py.
These two are the main code in our stepwise codes. Till step 6 you just see how a ppo (or pocar) works and how the CAT in DQN and without it works, now you are ready to experience the deep implementation of DQN_CAT with Huber and DQN_CAT by PPO under Huber loss functions.
I explained these 3 codes inside the paper and the ppo of the last code already covered in my implementation of gymnasium (I changed the environment of the google that was in https://github.com/google/ml-fairness-gym and in old python version to the updated environment of openAI Gymnasium).
So, codes of step 1 to 5 is just for setting up the CAT environment and ppo performance in CAt enviroment (dynamic in each moment for each agent). 

Below is an explanation for each .py file in the provided code document, formatted in paragraphs with added details on loss functions, reward structures, mathematical formulations, and key metrics. Each section focuses solely on the respective Python file, explaining its purpose, logic, mathematical underpinnings, and execution instructions. The files are part of a Computerized Adaptive Testing (CAT) project using reinforcement learning techniques like PPO, DQN, and POCAR.
README: CAT Agent Project - File Explanations
This document provides in-depth explanations for each .py file in a project focused on training reinforcement learning agents for Computerized
Adaptive Testing (CAT). CAT adapts item difficulty to a test-taker’s ability, estimated via Item Response Theory (IRT). The scripts implement
various RL algorithms (PPO, DQN, POCAR) and traditional methods, detailing their loss functions, rewards, metrics, and mathematical 
formulations. Each file is described below with its logic, mathematics, and how to run it.

train_agent.py
The train_agent.py script trains a Proximal Policy Optimization (PPO) reinforcement learning agent to optimize item selection in a CAT 
environment, leveraging the CATEnv class from cat_env.py. The environment simulates a CAT session where the agent estimates ability 
(theta_est) using a 3PL IRT model, terminating when the standard error (SE) drops below 0.3 or the item count reaches 30, with a minimum of 
10 items. The PPO agent, implemented via stable_baselines3, learns over 5000 timesteps to maximize a reward defined as 
reward = (theta_est - old_theta) + 0.5 * r, where old_theta is the previous estimate, and r is the binary response 
(1 if correct, 0 if incorrect). This reward encourages accurate ability updates (positive change in theta_est) 
and correct answers (bonus of 0.5). The observation space is a single float theta_est in [-4, 4], and the action space 
is discrete, selecting item indices from 0 to num_items-1. The PPO loss function combines a clipped surrogate objective for policy updates
, L^{CLIP} = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)], where r_t = π(a_t|s_t)/π_old(a_t|s_t) is the probability ratio, A_t is the 
advantage (estimated as reward plus discounted future value minus current value), and ε=0.2 is the clipping parameter, with an additional 
value loss L^{VF} = E[(V(s_t) - R_t)^2] (MSE) and entropy regularization (coefficient 0.0 here). Key metrics include total reward, 
final theta_est (compared to true ability 1.5), and SE, reflecting estimation accuracy and precision. After training, the agent is tested, 
printing these metrics and administered item indices. To run, Python 3.x is required with pip install stable-baselines3 gymnasium pandas 
numpy torch, and cat_env.py must be present. Execute with python train_agent.py, expecting output like Test completed with total 
reward=3.214 and final theta=1.487. This file depends on cat_env.py and should follow its preparation.

cat_env.py
The cat_env.py file defines the CATEnv class, a Gymnasium-style environment for CAT simulation, and a generate_item_bank function to create 
a synthetic item bank. The generate_item_bank function produces a DataFrame with num_items (default 100) items, each with parameters: 
difficulty b ~ N(0, 1) clipped to [-3, 3], discrimination a ~ logN(0, 0.5) clipped to [0, 3], guessing c ~ U(0.05, 0.3), slip ~ U(0.01, 0.3), 
and learning objectives as a JSON of 1-3 randomly chosen keys (lo1-lo4) with binary values. The CATEnv initializes with an item bank, true 
ability (default 0.0), and item limits (min 10, max 30). Its reset sets theta_est = 0, and step selects an item, computes response probability 
via 3PL IRT P(θ) = c + (1-c)/(1 + e^(-1.7a(θ-b))), simulates a response r ~ Bernoulli(P(θ)), updates theta_est using EAP, and calculates 
reward as reward = (theta_est - old_theta) + 0.5 * r. EAP estimation integrates over a grid θ ∈ [-4, 4] with nqt=31 points, 
using θ_est = Σ(θ * posterior) / Σ(posterior), where posterior is likelihood * prior, likelihood is Π[P(θ)^r * (1-P(θ))^(1-r)], 
and prior is N(0, 1). Fisher Information is I(θ) = 2.89 * a^2 * (1-c) / [(c + e^(1.7a(θ-b))) * (1 + e^(-1.7a(θ-b)))^2], and 
SE is 1 / √(ΣI(θ) + 1e-12). Termination occurs when steps ≥ 10 and (SE ≤ 0.3 or steps ≥ 30). No loss function exists here, as it’s an 
environment, but metrics include SE (precision) and theta_est accuracy. This file is imported by others (e.g., train_agent.py), 
requiring numpy, pandas, gymnasium, and scipy. Run it by ensuring it’s available in the same directory as dependent scripts, 
serving as a foundational dependency.

compare_cat_methods.py
The compare_cat_methods.py script simulates a traditional CAT session using Maximum Posterior Information (MPI) item selection, providing a 
baseline for RL methods. It uses generate_item_bank from cat_env.py to create an item bank and defines simulate_cat_session_MPI to start with 
the first item (lowest ID) and iteratively select items maximizing Fisher Information 
I(θ) = 2.89 * a^2 * (1-c) / [(c + e^(1.7a(θ-b))) * (1 + e^(-1.7a(θ-b)))^2)] at the current theta_est. Responses are simulated 
with P(θ) = c + (1-c)/(1 + e^(-1.7a(θ-b))), and theta_est is updated via EAP as in cat_env.py. No reward or loss function is defined, 
as this is a rule-based method, but the simulation tracks steps, SE, and theta_est accuracy against true ability (1.5). Termination 
mirrors CATEnv: min 10 items and (SE ≤ 0.3 or max 30 items). Outputs include true ability, theta_est, item count, SE, and item-response 
pairs, with metrics like estimation bias (theta_est - true_ability) and SE reflecting accuracy and precision. Running requires Python 3.x 
with pip install numpy pandas scipy, and cat_env.py must be present. Execute with python compare_cat_methods.py, producing output like 
Estimated Theta: 1.523, Standard Error: 0.287. This file depends on cat_env.py and should follow its availability.

pocar.py
The pocar.py script implements a Proximal Optimization with Clipped Advantage Regularization (POCAR) RL agent with fairness regularization 
for CAT. It includes a simplified CATEnv identical to cat_env.py, with reward reward = (theta_est - old_theta) + 0.5 * r. Fairness is 
introduced via fairness_metric(θ) = max(0, 1.6 - θ), penalizing theta_est below 1.6. Two neural networks drive the agent: PolicyNetwork 
outputs a softmax probability over items, and ValueNetwork predicts state value V(s). Training collects trajectories (5 episodes per 
iteration, 200 iterations, max 30 steps), computing advantages A = R + γ * V(s') - V(s) (γ=0.99), adjusted by 
fairness: A_fair = β0 * A + β1 * min(0, -Δ_t + ω) + β2 * min(0, Δ_t - Δ_{t+1}) if Δ_t > ω, where Δ_t = fairness_metric(θ_t), β0=β1=β2=1.0, 
and ω=0.5. The policy loss is L^{CLIP} = -E[min(r_t * A_fair, clip(r_t, 1-ε, 1+ε) * A_fair)] with r_t = π(a_t|s_t)/π_old(a_t|s_t) and ε=0.2, 
plus value loss L^{VF} = E[(V(s_t) - R_t)^2], combined as L = L^{CLIP} + 0.5 * L^{VF}. Metrics include total reward, final theta_est, and 
fairness violation (Δ_t). Outputs show iteration loss, theta_est, reward, and item details. Run with Python 3.x and pip install numpy pandas 
scipy torch, using python pocar.py. It’s standalone, producing logs like Iteration 100: Total Loss = 0.3241, Current Theta Estimate = 1.512.

QnetCAT1.py
The QnetCAT1.py script trains a Deep Q-Network (DQN) agent for CAT with a normal prior θ ~ N(0, 1). The Net class maps theta_est to Q-values 
for 500 items via a network with layers (1, 50, 30, 500). Training uses experience replay (memory size 1000), sampling batches (size 128) to 
update Q-values with loss L = E[(Q(s,a) - (R + γ * max(Q(s',a'))))^2] (MSE), where 
R = FI(θ) = D^2 * a^2 * (1-c) / [(c + e^(Da(θ-b))) * (1 + e^(-Da(θ-b)))^2], D=1, and γ=0.1. Actions follow an epsilon-greedy policy (ε=0.1), 
and the target network updates every 40 steps. Validation every 50 students tracks bias (theta_est - true_theta), 
RMSE (√E[(theta_est - true_theta)^2]), and MAE (E[|theta_est - true_theta|]), saving the best model. Testing outputs these metrics per step, 
saving to CSV. Requires Python 3.x, pip install numpy pandas scipy torch, and item_bank.csv/test_theta.csv in 
/content/drive/MyDrive/CAT_agent/. Run with python QnetCAT1.py, expecting logs like step 40, bias 0.123, rmse 0.345. It’s standalone but needs data files.

QnetCAT2.py
The QnetCAT2.py script replicates QnetCAT1.py with a uniform prior θ ~ U(-3, 3), adding a plot comparing normal vs. uniform prior performance. 
The DQN setup mirrors QnetCAT1.py: reward R = FI(θ), loss L = E[(Q(s,a) - (R + γ * max(Q(s',a'))))^2], γ=0.1, and metrics (bias, RMSE, MAE). 
The plotting section loads CSV outputs from both scripts, graphing metrics over 40 steps. Requires pip install numpy pandas scipy torch 
matplotlib and data files in /content/drive/MyDrive/CAT_agent/. Run with python QnetCAT2.py, producing logs and a plot. 
It’s standalone but benefits from QnetCAT1.py output for comparison.

QnetCAT_with_POCAR.py
The QnetCAT_with_POCAR.py script merges DQN with POCAR fairness to balance item usage across 50 items for 1000 students (20 steps each). 
The QNet maps theta_est to Q-values, with reward R = FI(θ) and fairness-adjusted advantage 
A = β0 * R + β1 * min(0, -Δ_t + ω) + β2 * min(0, Δ_t - Δ_{t+1}), where Δ_t = |item_count - target|, target = 400, β0=1.0, β1=β2=0.1, 
ω=2.0. Loss is L = E[(Q(s,a) - (A + γ * max(Q(s',a'))))^2], γ=0.9. Metrics include loss, reward, advantage, and item usage variance, 
logged to TensorBoard. Requires pip install numpy pandas scipy torch tensorboard. Run with python QnetCAT_with_POCAR.py, view logs with 
tensorboard --logdir=./logs/CAT_DQN. It’s standalone, outputting logs like loss 0.234, reward 1.543.

Citation:
Al Marjani, A., Garivier, A., and Proutiere, A. Navigating to the best policy in markov decision processes. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 25852–25864. Curran Associates, Inc.,2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/file/d9896106ca98d3d05b8cbdf4fd8b13a1-Paper.pdf.
Author et al. (2024). ***. Frontiers in Education,*, *.
Suilen, M., Simão, T. D., Parker, D., & Jansen, N. (2022). Robust anytime learning of Markov decision processes. In Advances in Neural Information Processing Systems (Vol. 35, pp. 28790–28802). Curran Associates, Inc.
Wang, P., Liu, H. & Xu, M. An adaptive testing item selection strategy via a deep reinforcement learning approach. Behav Res 56, 8695–8714 (2024). https://doi.org/10.3758/s13428-024-02498-x
Yu, E., Qin, Z., Lee, M. K., & Gao, S. (2022). Policy optimization with advantage regularization for long-term fairness in decision systems. In Advances in Neural Information Processing Systems (Vol. 35, pp. 8211–8213). Curran Associates, Inc.
