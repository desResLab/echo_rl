from Echo_class import Echo as echo1
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque

# Define separate folders for saving
penalty_folder = "penalty_vectors"
sonographer_folder = "sonographer_vectors"
model_folder = "models"

# Create all folders if they don't exist
os.makedirs(penalty_folder, exist_ok=True)
os.makedirs(sonographer_folder, exist_ok=True)
os.makedirs(model_folder, exist_ok=True)

# Constants
SEED = 42
GAMMA = 0.99
LEARNING_RATE = 0.025
LEARNING_RATE_MIN = 0.000025
MEMORY_SIZE = 100000
BATCH_SIZE = 32
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.99999
Optimization_Stride_size = 4
EXPLORATION_DECAY_Stride_size = 4
TRAINING_EPISODES = 30000
RATE_SONOGRAPHER_LEAVE_MAX = 1
RATE_SONOGRAPHER_LEAVE_MIN = 0.1
RATE_SONOGRAPHER_LEAVE_DECAY = (RATE_SONOGRAPHER_LEAVE_MAX - RATE_SONOGRAPHER_LEAVE_MIN) / 10000
GRADIENT_CLIP = 1.0
num_action = 3

# Set seeds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


class PrioritizedMemory:
    def __init__(self, capacity=MEMORY_SIZE, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

    def remember(self, state, action, reward, next_state, terminal):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.memory.append((state, action, reward, next_state, terminal))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        total = len(self.memory)
        priorities = np.array(self.priorities)
        probabilities = (priorities + self.epsilon) ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(total, batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        return samples, weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = error + self.epsilon


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.exploration_rate = EXPLORATION_MAX

        self.net = nn.Sequential(
            nn.Linear(13, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, num_action**6)
        )

        self.net.apply(self._init_weights)
        self.objective = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.95,
            patience=50, min_lr=LEARNING_RATE_MIN, verbose=True
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def act(self, state):
        action = []
        action_int = []
        fetal_pair = min(state[6], state[8])

        # Generate valid actions
        for i in range(min(fetal_pair + 1, state[0] + 1, num_action)):
            for j in range(min(fetal_pair + 1 - i, state[1] + 1, num_action)):
                for k in range(min(fetal_pair + 1 - i - j, state[2] + 1, num_action)):
                    for l in range(min(state[6] + state[7] + 1 - i - j - k,
                                       state[8] + state[9] + 1 - i - j - k,
                                       state[3] + 1, num_action)):
                        for m in range(min(state[6] + state[7] + 1 - i - j - k - l,
                                           state[8] + state[9] + 1 - i - j - k - l,
                                           state[4] + 1, num_action)):
                            for n in range(min(state[6] + state[7] + 1 - i - j - k - l - m,
                                               state[8] + state[9] + 1 - i - j - k - l - m,
                                               state[5] + 1, num_action)):
                                action.append([i, j, k, l, m, n])

        for a in action:
            action_int.append(a[0] * num_action**5 + a[1] * num_action**4 + a[2] * num_action**3 + a[3] * num_action**2 + a[4] * num_action + a[5])

        if np.random.rand() < self.exploration_rate:
            return random.choice(action_int)
        else:
            with torch.no_grad():
                self.net.eval()  # Set to evaluation mode
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.net(state_tensor).cpu().numpy().squeeze()
                self.net.train()  # Set back to training mode
                valid_q_values = [q_values[a] for a in action_int]
                return action_int[np.argmax(valid_q_values)]



    def forward(self, x):
        return self.net(x)


def optimize(policy_net, target_net, memory, all_steps):
    if len(memory.memory) > BATCH_SIZE:
        batch, weights, indices = memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, terminals = zip(*batch)

        # Convert list of arrays to a single array first
        states = np.array(states)
        state_batch = torch.FloatTensor(states).to(device)
        action_batch = torch.LongTensor(actions).to(device)
        reward_batch = torch.FloatTensor(rewards).to(device)
        next_states = np.array(next_states)
        next_state_batch = torch.FloatTensor(next_states).to(device)
        terminal_batch = torch.BoolTensor(terminals).to(device)
        weights = torch.FloatTensor(weights).to(device)

        current_q_values = policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))

        with torch.no_grad():
            next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = target_net(next_state_batch).gather(1, next_actions)
            next_q_values[terminal_batch] = 0.0
            expected_q_values = reward_batch.unsqueeze(1) + GAMMA * next_q_values

        loss = (weights.unsqueeze(1) * policy_net.objective(current_q_values, expected_q_values)).mean()

        if all_steps % Optimization_Stride_size == 0:
            policy_net.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), GRADIENT_CLIP)
            policy_net.optimizer.step()
            policy_net.scheduler.step(loss)

        # Get scalar TD errors for priorities
        td_errors = abs(current_q_values - expected_q_values).detach().cpu().numpy()
        td_errors = td_errors.flatten()  # Ensure 1D array of scalars
        memory.update_priorities(indices, td_errors)




time_start = '08:00'
time_close = '17:00'
num_fetal_room = 1
num_nonfetal_room = 6
num_sonographer_both = 4
num_sonographer_nonfetal = 2
time_sonographer_break = 15
rate_sonographer_leave = 0.1
rate_absence = 0.1
render_env = True

def echo_management():
    total_penalty_learning_vector = []
    rate_sonographer_leave_vector = []
    rate_sonographer_leave = RATE_SONOGRAPHER_LEAVE_MAX
    exploration_rate = EXPLORATION_MAX
    all_steps = 0

    env = echo1(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            render_env=render_env)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = PrioritizedMemory()

    for episode in range(1,TRAINING_EPISODES+1):
        state, info = env.reset(seed=episode, rate_sonographer_leave=rate_sonographer_leave)
        total_penalty = 0
        terminated = False

        while not terminated:
            all_steps += 1

            action = policy_net.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_penalty += -reward

            memory.remember(state, action, reward, next_state, terminated)

            if len(memory.memory) > (env.convert_to_step(time_close) + 120) * 10:
                if all_steps % Optimization_Stride_size == 0:
                    exploration_rate *= EXPLORATION_DECAY
                if all_steps == (env.convert_to_step(time_close) + 120) * 20000:
                    exploration_rate = EXPLORATION_MAX

                optimize(policy_net, target_net, memory, all_steps)

                # Hard update target network periodically
                if all_steps % (env.convert_to_step(time_close) + 120) == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            state = next_state


        policy_net.exploration_rate = exploration_rate

        # Store metrics
        total_penalty_learning_vector.append(total_penalty)
        rate_sonographer_leave_vector.append(rate_sonographer_leave)

        # Update sonographer leave rate
        if episode >= 10000:
            if rate_sonographer_leave > RATE_SONOGRAPHER_LEAVE_MIN:
                rate_sonographer_leave -= RATE_SONOGRAPHER_LEAVE_DECAY

        if episode % 100 == 0:
            # Save vectors
            penalty_path = os.path.join(penalty_folder, f"total_penalty_learning_vector_{episode}.npy")
            np.save(penalty_path, np.array(total_penalty_learning_vector))

            sonographer_path = os.path.join(sonographer_folder, f"rate_sonographer_leave_vector_{episode}.npy")
            np.save(sonographer_path, np.array(rate_sonographer_leave_vector))

            # Save periodic model checkpoints
            model_checkpoint_path = os.path.join(model_folder, f"policy_net_{episode}.pth")
            torch.save(policy_net.state_dict(), model_checkpoint_path)

    # Save final vectors
    final_penalty_path = os.path.join(penalty_folder, "total_penalty_learning_vector_final.npy")
    np.save(final_penalty_path, np.array(total_penalty_learning_vector))

    final_sonographer_path = os.path.join(sonographer_folder, "rate_sonographer_leave_vector_final.npy")
    np.save(final_sonographer_path, np.array(rate_sonographer_leave_vector))

    # Save final model
    final_model_path = os.path.join(model_folder, 'policy_net_final.pth')
    torch.save(policy_net.state_dict(), final_model_path)


if __name__ == "__main__":
    echo_management()