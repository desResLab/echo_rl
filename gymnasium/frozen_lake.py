import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Convert state to one-hot encoding
def one_hot_encode(state, size):
    one_hot = np.zeros(size)
    one_hot[state] = 1.0
    return one_hot

# Training function for DDQN
def train_ddqn(online_net, target_net, replay_buffer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, terminated in batch:
        state_tensor = torch.FloatTensor(one_hot_encode(state, state_size)).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(one_hot_encode(next_state, state_size)).unsqueeze(0)

        target = reward
        if not terminated:
            next_action = online_net(next_state_tensor).argmax().item()
            target += gamma * target_net(next_state_tensor)[0][next_action]

        # Update the online network
        target_f = online_net(state_tensor)
        target_f[0][action] = target

        loss = nn.MSELoss()(online_net(state_tensor), target_f.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Action selection with epsilon-greedy strategy
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Explore
    else:
        with torch.no_grad():
            return online_net(torch.FloatTensor(one_hot_encode(state, state_size)).unsqueeze(0)).argmax().item()  # Exploit

# Main function
if __name__ == "__main__":
    # Initialize environment and hyperparameters
    env = gym.make("FrozenLake-v1", render_mode="human")
    action_size = env.action_space.n
    state_size = env.observation_space.n

    online_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(online_net.state_dict())

    replay_buffer = deque(maxlen=2000)
    optimizer = optim.Adam(online_net.parameters(), lr=0.001)

    episodes = 1000
    batch_size = 32
    gamma = 0.99
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    min_epsilon = 0.01

    # Training loop
    for episode in range(episodes):
        observation, info = env.reset(seed=42)
        done = False
        total_reward = 0

        while not done:
            action = select_action(observation, epsilon)

            # Step through the environment
            next_state, reward, terminated, truncated, info = env.step(action)
            if terminated and reward == 0:
                reward = -100.0  # Assign penalty for failing
            elif terminated and reward == 1:
                reward = 100.0
            else:
                reward = -1

            total_reward += reward
            print(f"Action: {action}, Reward: {reward}, Next State: {next_state}, Terminated: {terminated}")
            # Store experience in replay buffer
            replay_buffer.append((observation, action, reward, next_state, terminated))
            observation = next_state

            # Train DDQN
            train_ddqn(online_net, target_net, replay_buffer, batch_size, gamma)

            if terminated or truncated:
                break  # End episode if terminated or truncated

        # Periodically update the target network
        if episode % 10 == 0:
            target_net.load_state_dict(online_net.state_dict())

        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    env.close()
