import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from Echo_trivial import Echo
from Echo_class import Echo as Echo1
import numpy as np
import random
import torch
import torch.nn as nn
import os

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
num_action = 3

sim = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            render_env=render_env)
sim1 = Echo1(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            render_env=render_env)

rows = sim.convert_to_step(time_close) +121
ontime_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
late_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
early_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
fetal_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
nonfetal_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
available_fetal_room = [0 for _ in range(rows)]
available_nonfetal_room = [0 for _ in range(rows)]
available_fetal_sonographer = [0 for _ in range(rows)]
available_nonfetal_sonographer = [0 for _ in range(rows)]
current_time = sim.convert_to_step(time_start)
penalty = [[0] * rows for _ in range(15)]

class DQN(nn.Module):
    def __init__(self, load_path):
        super(DQN, self).__init__()
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
        # Load the model parameters if a path is provided
        self.load_model('models/policy_net_final.pth')



    def load_model(self, path):
        """Load pre-trained model parameters from the given path, adjusting keys if necessary."""
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        # Remove the "net." prefix from keys
        new_state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}

        self.net.load_state_dict(new_state_dict)
        print(f"Model loaded from {path}")

    def act(self, state):
        """Select action based on current state using the policy network."""

        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        action = []
        action_int = []
        fetal_pair = min(state[6], state[8])

        # Generate all possible valid actions
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

        # Convert valid actions to integer representations
        for a in action:
            action_int.append(a[0] * num_action**5 + a[1] * num_action**4 + a[2] * num_action**3 + a[3] * num_action**2 + a[4] * num_action + a[5])

        # Get predicted Q values and select best action
        with torch.no_grad():
            q_values = self.net(state_tensor).squeeze(0).numpy()
            if len(action_int) == 0:
                return 0  # Default action if no valid actions found
            valid_q_values = [q_values[a] for a in action_int]
            best_action_idx = np.argmax(valid_q_values)
            action = action_int[best_action_idx]

        return action

def calculate_accommodation(action_int):
    """Decomposes the action_int into six categories of accommodations."""
    action = [0] * 6
    for i in range(6):
        divisor = num_action ** (5 - i)
        action[i] = (action_int // divisor)
        action_int %= divisor
    return action


policy_net = DQN(load_path='policy_net.pth')
policy_net.eval()
for iteration in range(365):
    for policy in [0,1,2,3,4]:
        random.seed(iteration)
        np.random.seed(iteration)
        if policy == 0:
            state, info = sim1.reset(rate_sonographer_leave=0.1)  # Reset simulation for each policy
        else:
            sim.reset(rate_sonographer_leave=0.1)
        if policy == 5:
            sim._make_reservations(0,0)
        if policy == 6:
            sim._make_reservations(0,0.25)
        if policy == 7:
            sim._make_reservations(0,0.5)
        if policy == 8:
            sim._make_reservations(0,0.75)
        if policy == 9:
            sim._make_reservations(0,1)
        if policy == 10:
            sim._make_reservations(1,0)
        if policy == 11:
            sim._make_reservations(1,0.25)
        if policy == 12:
            sim._make_reservations(1,0.5)
        if policy == 13:
            sim._make_reservations(1,0.75)
        if policy == 14:
            sim._make_reservations(1,1)
        if policy == 15:
            sim._make_reservations(0,0)
        if policy == 16:
            sim._make_reservations(0,0.25)
        if policy == 17:
            sim._make_reservations(0,0.5)
        if policy == 18:
            sim._make_reservations(0,0.75)
        if policy == 19:
            sim._make_reservations(0,1)
        if policy == 20:
            sim._make_reservations(1,0)
        if policy == 21:
            sim._make_reservations(1,0.25)
        if policy == 22:
            sim._make_reservations(1,0.5)
        if policy == 23:
            sim._make_reservations(1,0.75)
        if policy == 24:
            sim._make_reservations(1,1)
        current_time = sim.convert_to_step('08:00')  # Convert the time directly to the desired format
        if policy ==0:
            penalty[policy][current_time] += sim1._penalty_calculation() / 365
        else:
            penalty[policy][current_time] += sim._penalty_calculation() / 365
        end_time = sim.convert_to_step(time_close) + 120


        while current_time <= end_time:
            if policy == 0:
                policy_net.eval()
                action = policy_net.act(state)
                state_next, reward, terminated, truncated, info = sim1.step(action)
                state = state_next
            elif policy < 5:
                state_next, action, reward, terminated, truncated, info = sim.step(policy)
            elif policy < 15:
                state_next, action, reward, terminated, truncated, info = sim.step(5)
            else:
                state_next, action, reward, terminated, truncated, info = sim.step(6)
            penalty[policy][current_time] += (-1)*reward
            current_time += 1


start_time = datetime.strptime('08:00', '%H:%M')
time_range = [start_time + timedelta(minutes=i) for i in range(len(penalty[1]))]  # Adjust based on number of policies

# Plotting
plt.figure(figsize=(12, 6))
# Define a list of colors
colors = [
    # Primary and Basic Colors
    'blue', 'green', 'red', 'yellow', 'black',

    # Shades and Tints
    'pink', 'brown', 'gray', 'lightblue', 'olive',

    # Unique Tones
    'purple', 'orange', 'cyan', 'magenta', 'teal', 'turquoise',

    # Deep and Dark Colors
    'maroon', 'navy', 'lime', 'indigo', 'gold',

    # Light and Neutral Colors
    'violet', 'silver', 'coral', 'beige'
]

for i in [0,1,2,3,4]:
    if i == 0:
        plt.plot(time_range, penalty[i], label=f'Learning policy', color=colors[i])
    elif i < 5:
        plt.plot(time_range, penalty[i], label=f'Policy {i}', color=colors[i])
    elif i == 5:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0)', color=colors[i])
    elif i == 6:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0.25)', color=colors[i])
    elif i == 7:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0.5)', color=colors[i])
    elif i == 8:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0.75)', color=colors[i])
    elif i == 9:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=1)', color=colors[i])
    elif i == 10:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0)', color=colors[i])
    elif i == 11:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0.25)', color=colors[i])
    elif i == 12:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0.5)', color=colors[i])
    elif i == 13:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0.75)', color=colors[i])
    elif i == 14:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=1)', color=colors[i])
    elif i == 15:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=0, beta=0)', color=colors[i])
    elif i == 16:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=0, beta=0.25)', color=colors[i])
    elif i == 17:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=0, beta=0.5)', color=colors[i])
    elif i == 18:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=0, beta=0.75)', color=colors[i])
    elif i == 19:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=0, beta=1)', color=colors[i])
    elif i == 20:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=1, beta=0)', color=colors[i])
    elif i == 21:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=1, beta=0.25)', color=colors[i])
    elif i == 22:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=1, beta=0.5)', color=colors[i])
    elif i == 23:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=1, beta=0.75)', color=colors[i])
    elif i == 24:
        plt.plot(time_range, penalty[i], label='Policy 6 (alpha=1, beta=1)', color=colors[i])

fs=20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Penalty', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.legend(fontsize=fs-7)
plt.title('Penalty for Each Policy', fontsize=fs)
# Set x-axis ticks to represent time in "00:00" format
plt.xticks(time_range[::60], [time.strftime('%H:%M') for time in time_range[::60]], rotation=45)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'penalty plot (1-4).pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()