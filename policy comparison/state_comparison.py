from Echo_trivial import Echo as echo1
from Echo_class import Echo as echo2
import random
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import copy

num_action = 3
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
        self.load_model(load_path)



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


# Initialize parameters
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

# Initialize model and environments
policy_net = DQN(load_path='models/policy_net_final.pth')
policy_net.eval()

echo1 = echo1(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
              num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
              num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
              rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
              render_env=render_env)
echo2 = echo2(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
              num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
              num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
              rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
              render_env=render_env)



for i in range(1,100):
    # Reset the lists for this iteration
    different_states = []
    different_actions_1 = []
    different_actions_2 = []
    different_penalties_1 = []
    different_penalties_2 = []
    different_times = []
    random.seed(i)
    np.random.seed(i)
    echo1.reset(seed = i, options = None, rate_sonographer_leave = 0.1)
    echo2.reset(seed = i, options = None, rate_sonographer_leave = 0.1)

    while echo1.env.now <= echo1.convert_to_step(time_close):

        state = [
            echo1._ontime_waiting_fetal_patients(),
            echo1._late_waiting_fetal_patients(),
            echo1._early_waiting_fetal_patients(),
            echo1._ontime_waiting_nonfetal_patients(),
            echo1._late_waiting_nonfetal_patients(),
            echo1._early_waiting_nonfetal_patients(),
            echo1._available_fetal_sonograpphers(),
            echo1._available_nonfetal_sonograpphers(),
            echo1._available_fetal_rooms(),
            echo1._available_nonfetal_rooms(),
            echo1._onbreak_fetal_sonograpphers(),
            echo1._onbreak_nonfetal_sonograpphers(),
            echo1.env.now
        ]

        echo2.state = copy.deepcopy(echo1.state)

        # Get actions from both policies
        state_next, action_1, reward_1, terminated, truncated, info = echo1.step(2)
        action_2_int = policy_net.act(state)
        action_2 = calculate_accommodation(action_2_int)

        # Compare the actions after converting action_1 to a list

        # Only store if actions are different
        if action_1 != action_2:
            current_time = (datetime.strptime(time_start, '%H:%M') +
                            timedelta(minutes=echo1.env.now-1)).strftime('%H:%M')
            different_times.append(current_time)
            different_states.append(state)
            different_actions_1.append(action_1)
            different_actions_2.append(action_2)
            different_penalties_1.append((-1) * reward_1)

            # Get reward for policy 2
            state_next, reward_2, terminated, truncated, info = echo2.step(action_2_int)  # Pass the integer action
            different_penalties_2.append((-1) * reward_2)
        else:
            # Still need to step echo2 to maintain synchronization
            state_next, reward_2, terminated, truncated, info = echo2.step(action_2_int)  # Pass the integer action
    # Convert times to datetime objects and sort
    # Extract the sorting key based on the fifth element of different_states
    sorted_indices = sorted(range(len(different_states)), key=lambda i: different_states[i][12])

    # Sort all related lists by the sorted indices
    different_states = [different_states[i] for i in sorted_indices]
    different_times = [different_times[i] for i in sorted_indices]
    print(different_times)
    different_actions_1 = [different_actions_1[i] for i in sorted_indices]
    different_actions_2 = [different_actions_2[i] for i in sorted_indices]
    different_penalties_1 = [different_penalties_1[i] for i in sorted_indices]
    different_penalties_2 = [different_penalties_2[i] for i in sorted_indices]

    # Plotting
    state_labels = [
        r"$W_{f,t}$", r"$W_{f,l}$", r"$W_{f,e}$",
        r"$W_{n,t}$", r"$W_{n,l}$", r"$W_{n,e}$",
        r"$S_b$", r"$S_n$", r"$R_b$", r"$R_n$",
        r"$L_f$", r"$L_n$", r"$T$"
    ]

    action_labels = [
        r"$W_{f,t}$", r"$W_{f,l}$", r"$W_{f,e}$",
        r"$W_{n,t}$", r"$W_{n,l}$", r"$W_{n,e}$"
    ]

    # Create figure with three subplots
    fig = plt.figure(figsize=(12, 10))  # Reduced overall height
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 0.5, 1])  # Reduced heights for middle and bottom subplots

    # Top Plot: Penalty Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    x_positions = range(len(different_times))
    ax1.plot(x_positions, different_penalties_1, 'o--', label="Penalty of Pre-defined policy 2", color='blue')
    ax1.plot(x_positions, different_penalties_2, 'o--', label="Penalty of Learning policy", color='red')

    ax1.set_xticks([])
    ax1.set_xlim(-0.5, len(different_times) - 0.5)
    ax1.set_ylabel("Penalty", fontsize=16)
    ax1.set_title("Penalty Comparison (Different Actions Only)", fontsize=16, pad=15)
    ax1.legend(loc="upper right", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.yaxis.set_label_coords(-0.05, 0.5)

# Middle Plot: Action Grid
    ax2 = fig.add_subplot(gs[1, 0])
    for x in range(len(different_actions_2)):
        for y in range(len(different_actions_2[x])):
            ax2.text(x, y, str(different_actions_2[x][y]), ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black"))

    ax2.set_xticks([])
    ax2.set_yticks(range(len(action_labels)))
    ax2.set_yticklabels(action_labels, fontsize=10)
    ax2.set_ylabel("Actions", fontsize=16)
    ax2.set_title("Learning Policy Actions (Different Actions Only)", fontsize=16, pad=15)  # Adjusted fontsize
    ax2.grid(False)
    ax2.yaxis.set_label_coords(-0.05, 0.5)

    # Bottom Plot: State Grid
    ax3 = fig.add_subplot(gs[2, 0])
    for x, time in enumerate(different_times):
        for y, label in enumerate(state_labels):
            ax3.text(x, y, different_states[x][y], ha='center', va='center', fontsize=12,
                     bbox=dict(boxstyle="round", facecolor="lightgrey", edgecolor="black"))

    ax3.set_xticks(range(len(different_times)))
    ax3.set_xticklabels(different_times, fontsize=10, rotation=45, ha='right')
    ax3.set_yticks(range(len(state_labels)))
    ax3.set_yticklabels(state_labels, fontsize=10)

    ax3.set_xlabel("Time", fontsize=16)
    ax3.set_ylabel("States", fontsize=16)
    ax3.set_title("States (Different Actions Only)", fontsize=16, pad=15)  # Adjusted fontsize
    ax3.grid(False)
    ax3.yaxis.set_label_coords(-0.05, 0.5)

    # Ensure all subplots have the same x-axis limits
    ax1.set_xlim(-0.5, len(different_times) - 0.5)
    ax2.set_xlim(-0.5, len(different_times) - 0.5)
    ax3.set_xlim(-0.5, len(different_times) - 0.5)

    # Adjust layout
    plt.subplots_adjust(hspace=0.8, left=0.1)
    plt.tight_layout()

    plt.savefig(f"states/case_{i}.pdf")  # Save each case in a separate PDF file
    plt.close(fig)

    print(f"Case {i} saved successfully.")

