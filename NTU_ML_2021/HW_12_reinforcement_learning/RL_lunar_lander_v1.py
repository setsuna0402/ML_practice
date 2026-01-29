'''
Propose:
Train a reinforcement learning agent using the policy gradient method on the Lunar Lander environment from OpenAI Gym.
Author: Dr. Ka Hou Leong
Date: 28/1/2026
Verion: 0.1
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

# Define the agent network
class PolicyNetwork(nn.Module):
    # In the Lunar Lander environment, the state space has 8 dimensions and action space has 4 discrete actions
    def __init__(self, state_space=8, action_space=4):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        # dropout layer with p=0.25
        self.dropout = nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(64, action_space)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=-1)
    
# Define the Policy Gradient Agent
class PolicyGradientAgent():
    def __init__(self, network, learning_rate=0.001, gamma=0.99):
        self.network = network
        # self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.log_probs = []
        self.rewards = []

    def forward(self, state):
        return self.network(state)
    
    def learn_trivial(self, log_probs, rewards):
        ### parameters of action don't affect the prob(s_t+1|s_t, a_t) ###
        ### So, we only include prob(a_t|s_t) in the calculation of loss ###
        # Here, we consider the total reward for each trajectory: 
        # expectation value of reward = sum(trajectory's total reward * probability of the trajectory) / N
        # This is the worst method, because it treats all actions in the trajectory equally.

        # log_probs: 2D list of log probabilities of actions taken [i][j]: i-th trajectory, j-th action
        # rewards: 2D list of rewards for each action taken [i][j]: i-th trajectory, j-th action
        losses = []
        for ep_logps, ep_rewards in zip(log_probs, rewards):
            ep_logps = torch.stack(ep_logps)  # list [tensor_scale_a, tensor_scale_b, ..., tensor_scale_c] -> 1D Tensor [a, b, c, ..., ]
            ep_total_reward = sum(ep_rewards)  # scalar
            ep_returns = torch.ones_like(ep_logps) * ep_total_reward  # 1D Tensor [R, R, R, ..., R]
            losses.append(-(ep_logps * ep_returns).sum())
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self, log_probs, rewards):
        # This is the default learn function
        # log_probs and rewards are both 2D lists (from all trajectories concatenated)
        # Convert lists to flat tensors
        log_probs_flat = torch.cat([torch.stack(ep) for ep in log_probs]) 
        rewards_flat   = torch.tensor([r for ep in rewards for r in ep], dtype=torch.float32)
        # Normalise rewards (for the default learn function)
        rewards_flat = (rewards_flat - rewards_flat.mean()) / (rewards_flat.std() + 1e-9)
        loss = -torch.sum(log_probs_flat * rewards_flat) / EPISODE_PER_BATCH
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)  # Create a categorical distribution over the action probabilities
        action = action_dist.sample()  # Sample an action from the distribution
        log_prob = action_dist.log_prob(action) # Get the log probability of the sampled action
        return action.item(), log_prob

    def save(self, PATH): # You should not revise this
        Agent_Dict = {
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict()
        }
        torch.save(Agent_Dict, PATH)

    def load(self, PATH): # You should not revise this
        checkpoint = torch.load(PATH)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])




# Define environment
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")

fix(seed)
# Create the agent
network = PolicyNetwork()
agent = PolicyGradientAgent(network)

#Reset the environment to generate the first observation
observation, info = env.reset(seed=seed)
env.action_space.seed(seed)

agent.network.train()  # agent in training mode
EPISODE_PER_BATCH = 10  # Update the agent network every 5 episodes
NUM_BATCH = 400        # Update the agent network for 400 times in total

avg_total_rewards, avg_final_rewards = [], []

start_time = time.time()
prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []   # 2D lists [i][j]: i-th episode, j-th step
    total_rewards, final_rewards = [], []  # 1D lists for statistics

    # Collect data from multiple episodes
    for episode in range(EPISODE_PER_BATCH):
        # print("batch ", batch, " episode ", episode)
        log_probs.append([])  # log_probs for this episode
        # rewards.append([])    # rewards for this episode
        state, _ = env.reset()
        total_reward = 0.0
        total_step = 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state) # a_t , log_prob(a_t|s_t)
            next_state, reward, terminated, truncated, info = env.step(action)

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
    # rewards = (rewards - np.mean(rewards_flat)) / (np.std(rewards_flat) + 1e-9) # Normalize rewards
    agent.learn(log_probs, rewards)
    '''
    # For learn_trivial 
    # agent.learn_trivial(log_probs, rewards)  # No reward normalization in this method
    '''
    # print("logs prob looks like ", torch.stack(log_probs).size())
    # print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())

env.close()
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

end = time.time()
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()

plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()


