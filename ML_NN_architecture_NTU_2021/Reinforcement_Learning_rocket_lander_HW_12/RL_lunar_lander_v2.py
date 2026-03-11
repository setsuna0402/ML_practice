'''
Propose:
Train a reinforcement learning agent using the policy gradient method on the Lunar Lander environment from OpenAI Gym.
In this version, we consider a critic to estimate the value function. We use a neural network to approximate the value function.
The estimation of the value function is done by Monte Carlo method.
Author: Dr. Ka Hou Leong
Date: 28/1/2026
Version: 0.2
ML library: PyTorch
'''
import gymnasium as gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from PolicyAgent_Class import *

# Set random seeds for reproducibility
seed = 543 # Do not change this
def fix(seed):
  # env.seed(seed)
  # env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.enabled = True


save_path = "lunar_lander_actor_critic_mc_method"
# Define environment
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")

fix(seed)
# Create the agent
actor_network = PolicyNetwork()
value_network = ValueNetwork()
agent = ActorCriticAgent(actor_network, value_network, gamma=0.995)

#Reset the environment to generate the first observation
observation, info = env.reset(seed=seed)
env.action_space.seed(seed)

agent.actor_network.train()  # agent in training mode
agent.value_network.train()  # agent in training mode
EPISODE_PER_BATCH = 10  # Update the agent network every 10 episodes
NUM_BATCH = 800        # Update the agent network for 800 times in total

avg_total_rewards, avg_final_rewards = [], []

start_time = time.time()
prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards, states = [], [], []  # 2D lists [i][j]: i-th episode, j-th step
    total_rewards, final_rewards = [], []    # 1D lists for statistics

    # Collect data from multiple episodes
    for episode in range(EPISODE_PER_BATCH):
        # print("batch ", batch, " episode ", episode)
        log_probs.append([])  # log_probs for this episode
        states.append([])     # states for this episode
        # rewards.append([])    # rewards for this episode
        state, _ = env.reset()
        total_reward = 0.0
        total_step = 0
        seq_rewards = []
        while True:
            action, log_prob = agent.sample(state) # a_t , log_prob(a_t|s_t)
            next_state, reward, terminated, truncated, info = env.step(action)
            states[episode].append(state)  # [s_1, s_2, s_3, ...., s_t]
            log_probs[episode].append(log_prob) # [log(a_1|s_1), log(a_2|s_2), ...., log(a_t|s_t)]
            seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            # rewards[episode].append(reward)
            # simple implementation of rewards: 
            # rewards[i][j] : reward at step j in episode i
            # Like: a_1, a_2, a_3, ......
            #       r_1, r_2, r_3, ......
            # For different implementations of rewards:
            # beginner: keep the current implementation

            # medium: change to accumulative decaying reward
            # a_1,                        a_2,                           a_3 ......
            # r_1+0.99*r_2+0.99^2*r_3+......, r_2+0.99*r_3+0.99^2*r_4+...... ,r_3+0.99*r_4+0.99^2*r_5+ ......

            # boss: implement Deep Q Network (DQN)

            if terminated or truncated:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                break
        temp_list = []
        temp_reward = 0.0
        # Compute discounted rewards for the episode
        for i in reversed(seq_rewards):
            temp_reward = i + agent.gamma * temp_reward
            temp_list.insert(0, temp_reward)
        rewards.append(temp_list)

    # print(f"rewards looks like ", np.shape(rewards))
    # print(f"log_probs looks like ", np.shape(log_probs))

    # Record average rewards for monitoring
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # Update the agent network using the collected data
    agent.learn_actor_critic_mc(log_probs, rewards, states)
    '''
    # For learn_trivial 
    # agent.learn_trivial(log_probs, rewards)  # No reward normalization in this method
    '''
    # print("logs prob looks like ", torch.stack(log_probs).size())
    # print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

env.close()
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")
# Save agent's network weights
agent.save("./{}_num_batch_{}_episodes_{}_gamma_{}.pth".format(save_path, NUM_BATCH, EPISODE_PER_BATCH, agent.gamma))

plt.plot(avg_total_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Total Reward")
plt.title("Total Rewards")
plt.grid()
plt.savefig("./lunar_lander_total_rewards_num_batch_{}_episodes_{}_gamma_{}.png".format(NUM_BATCH, EPISODE_PER_BATCH, agent.gamma))
plt.close()
# plt.show()

plt.plot(avg_final_rewards)
plt.xlabel("Episode")
plt.ylabel("Average Final Reward")
plt.title("Final Rewards")
plt.grid()
plt.savefig("./lunar_lander_final_rewards_num_batch_{}_episodes_{}_gamma_{}.png".format(NUM_BATCH, EPISODE_PER_BATCH, agent.gamma))
plt.close()
# plt.show()


