import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque


# Discount rate of future rewards
GAMMA = 0.99
# Learning rate for neural network
LEARNING_RATE = 0.05
LEARNING_RATE_MIN = 0.00003
# Maximum number of game steps (state, action, reward, next state) to keep
MEMORY_SIZE = 100000
# Sample batch size for policy network update
BATCH_SIZE = 32
# Exploration rate (epislon) is probability of choosing a random action·
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
# Learning rate decay
LEARNING_RATE_DECAY = 0.99
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.99
# Stride size
Stride_size = 4
# Training episodes
TRAINING_EPISODES = 20000


class DQN(nn.Module):
    """Deep Q Network. Used for both policy (action) and target (Q) networks."""

    def __init__(self):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = EXPLORATION_MAX
        self.Stride_size = Stride_size
        self.learning_rate = LEARNING_RATE

        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 64)
        )

        self.net.apply(self._init_weights)

        # Set loss function and optimizer
        self.objective = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=LEARNING_RATE)

    def _init_weights(self, m):
        """Initialize weights and biases to 0."""
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)    # Set biases to 0


    def act(self, state):
        action = []
        action_int = []
        fetal_pair = min(state[6], state[8])
        # Generate all possible valid actions
        for i in range(min(fetal_pair + 1, state[0] + 1, 2)):
            for j in range(min(fetal_pair + 1 - i, state[1] + 1, 2)):
                for k in range(min(fetal_pair + 1 - i - j, state[2] + 1, 2)):
                    for l in range(min(state[6] + state[7] + 1 - i - j - k,
                                       state[8] + state[9] + 1 - i - j - k,
                                       state[3] + 1, 2)):
                        for m in range(min(state[6] + state[7] + 1 - i - j - k - l,
                                           state[8] + state[9] + 1 - i - j - k - l,
                                           state[4] + 1, 2)):
                            for n in range(min(state[6] + state[7] + 1 - i - j - k - l - m,
                                           state[8] + state[9] + 1 - i - j - k - l - m,
                                           state[5] + 1, 2)):
                                action.append([i, j, k, l, m, n])

        # Convert valid actions to integer representations
        for a in action:
            action_int.append(a[0] * 32 + a[1] * 16 + a[2] * 8 + a[3] * 4 + a[4] * 2 + a[5])

        # Act randomly if random number < exploration rate
        if np.random.rand() < self.exploration_rate:
            # Select a random action from the limited action set
            action = random.choice(action_int)
        else:
            # Get predicted Q values of all actions
            q_values = self.net(torch.FloatTensor(state)).detach().numpy()

            # Select the Q values only for the valid actions
            valid_q_values = [q_values[a] for a in action_int]

            # Get the index of the action with the highest Q value in valid actions
            best_action_idx = np.argmax(valid_q_values)

            # Map the index back to the original action from action_int
            action = action_int[best_action_idx]

        return action

    def forward(self, x):
        """Forward pass through network"""
        return self.net(x)

    ################################################################################
    #                    4 Define policy net training function                     #
    ################################################################################

    def optimize(policy_net, target_net, memory, learning_rate):

       for param_group in policy_net.optimizer.param_groups:
           param_group['lr'] = learning_rate

       # Do not try to train model if memory is less than reqired batch size
       if len(memory) < BATCH_SIZE:
           return

       # Reduce exploration rate (exploration rate is stored in policy net)
       policy_net.exploration_rate *= EXPLORATION_DECAY
       policy_net.exploration_rate = max(EXPLORATION_MIN,
                                        policy_net.exploration_rate)
       # Sample a random batch from memory
       batch = random.sample(memory, BATCH_SIZE)
       for state, action, reward, state_next, terminal in batch:

           state_action_values = policy_net(torch.FloatTensor(state))

           # Get target Q for policy net update

           if not terminal:
               # For non-terminal actions get Q from policy net
               expected_state_action_values = policy_net(torch.FloatTensor(state))
               # Detach next state values from gradients to prevent updates
               expected_state_action_values = expected_state_action_values.detach()
               # Get next state action with best Q from the policy net (double DQN)
               policy_next_state_values = policy_net(torch.FloatTensor(state_next))
               policy_next_state_values = policy_next_state_values.detach()
               best_action = np.argmax(policy_next_state_values.numpy())
               # Get target net next state
               next_state_action_values = target_net(torch.FloatTensor(state_next))
               # Use detach again to prevent target net gradients being updated
               next_state_action_values = next_state_action_values.detach()
               best_next_q = next_state_action_values[best_action]
               updated_q = reward + (GAMMA * best_next_q)
               expected_state_action_values[action] +=  learning_rate*(updated_q - expected_state_action_values[action])
           else:
               # For terminal actions Q = reward (-1)
               expected_state_action_values = policy_net(torch.FloatTensor(state))
               # Detach values from gradients to prevent gradient update
               expected_state_action_values = expected_state_action_values.detach()
               # Set Q for all actions to reward (-1)
               expected_state_action_values[action] = reward

           # Set net to training mode
           policy_net.train()
           # Reset net gradients
           policy_net.optimizer.zero_grad()
           # calculate loss
           loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
           # Backpropogate loss
           loss_v.backward()
           # Update network gradients
           policy_net.optimizer.step()


       return