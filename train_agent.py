import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from cat_env import CATEnv, generate_item_bank

def make_env():
    # Generate the item bank
    item_bank_df = generate_item_bank(num_items=100, seed=123)
    # Fix the agent's true ability at +1.5 for demonstration
    env = CATEnv(item_bank_df=item_bank_df, agent_true_ability=1.5, min_items=10, max_items=30)
    return env



def main():
    vec_env = DummyVecEnv([make_env])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        n_steps=64,
        batch_size=32,
        learning_rate=1e-3,
        gamma=0.99,
        ent_coef=0.0,
        device="cpu"  # Force CPU usage
    )

    # Train the model
    model.learn(total_timesteps=5000)

    # Test the learned policy on a fresh environment
    env = make_env()
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Test completed with total reward={total_reward:.3f} and final theta={obs[0]:.3f}")
    print("RL Agent Administered Item Indices:", env.administered_indices)

    # Optionally, if you recorded detailed item info:
    print("RL Agent Administered Items Details:")
    for detail in env.administered_item_details:
        print(detail)

if __name__ == '__main__':
    main()
