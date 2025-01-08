from Echo_class import Echo as echo1
from Echo_trivial import Echo as echo2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import os

num_action = 3
# Set the seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
EPISODES = 10000
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
                                            state[4] + 1, 2)):
                            for n in range(min(state[6] + state[7] + 1 - i - j - k - l - m,
                                                state[8] + state[9] + 1 - i - j - k - l - m,
                                                state[5] + 1, num_action)):
                                action.append([i, j, k, l, m, n])

        # Convert valid actions to integer representations
        for a in action:
            action_int.append(a[0] * num_action**5 + a[1] * num_action**4 + a[2] * num_action**3 + a[3] * num_action**2 + a[4] * num_action**1 + a[5])

        # Get predicted Q values and select best action
        with torch.no_grad():
            q_values = self.net(state_tensor).squeeze(0).numpy()
            if len(action_int) == 0:
                return 0  # Default action if no valid actions found
            valid_q_values = [q_values[a] for a in action_int]
            best_action_idx = np.argmax(valid_q_values)
            action = action_int[best_action_idx]

        return action

    def forward(self, x):
        """Forward pass through network"""
        return self.net(x)


def plot_difference_average(total_penalty_trivial_vector, total_penalty_learning_vector):
    # Convert to NumPy arrays for element-wise subtraction
    total_penalty_trivial_vector = np.array(total_penalty_trivial_vector)
    total_penalty_learning_vector = np.array(total_penalty_learning_vector)

    # Ensure vectors are the same length (optional step if necessary)
    min_len = min(len(total_penalty_trivial_vector), len(total_penalty_learning_vector))
    total_penalty_trivial_vector = total_penalty_trivial_vector[:min_len]
    total_penalty_learning_vector = total_penalty_learning_vector[:min_len]
    trivial_cumulative_sum = np.cumsum(total_penalty_trivial_vector)
    learning_cumulative_sum = np.cumsum(total_penalty_learning_vector)

    # Compute the reward difference
    trivial_progressive_mean = trivial_cumulative_sum / (np.arange(1, len(total_penalty_learning_vector) + 1))
    learning_progressive_mean = learning_cumulative_sum / (np.arange(1, len(total_penalty_learning_vector) + 1))

    # Create a range for the x-axis that starts from 1 and goes to the length of the vectors
    x_values = range(1, len(total_penalty_trivial_vector) + 1)
    fs = 20
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text', usetex=True)
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.xticks([EPISODES])
    plt.plot(trivial_progressive_mean, label='Mean of Trivial Policy daily Penalty', color='red')
    plt.plot(learning_progressive_mean, label='Mean of Learning Policy daily Penalty', color='green')

    # Adding titles and labels
    plt.title('Comparison of Total Penalties: Trivial vs Learning Policies', fontsize=fs)
    plt.xlabel('Time (days)', fontsize=fs)
    plt.ylabel('Penalty Average', fontsize=fs)
    plt.tick_params(axis='both', which='both', labelsize=fs - 2)


    # Display the plot
    plt.grid(True)
    # Add a legend to differentiate the two curves
    plt.legend(fontsize=fs-5)

    # Display the plot
    plt.tight_layout()
    pdf_path = os.path.join('penalty_average_two_policies.pdf')
    plt.savefig(pdf_path, format='pdf')
    plt.close()


def plot_results(total_penalty_trivial_vector, total_penalty_learning_vector):
    # Convert to NumPy arrays
    total_penalty_trivial_vector = np.array(total_penalty_trivial_vector)
    total_penalty_learning_vector = np.array(total_penalty_learning_vector)

    # Font and size settings
    fs = 20
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text', usetex=True)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot histograms for the two distributions
    plt.hist(
        total_penalty_trivial_vector, bins=30, density=True, alpha=0.5,
        label="Trivial Policy Penalty", color='blue', edgecolor='black'
    )
    plt.hist(
        total_penalty_learning_vector, bins=30, density=True, alpha=0.5,
        label="Learning Policy Penalty", color='green', edgecolor='black'
    )

    # Add labels and title
    plt.xlabel("Penalty Value", fontsize=fs)
    plt.ylabel("Density", fontsize=fs)
    plt.title("Distribution of Penalty: Trivial vs Learning Policy", fontsize=fs)

    # Add a legend to differentiate the two distributions
    plt.legend(fontsize=fs - 5)

    # Display the plot
    plt.tight_layout()
    plt.grid(True)
    pdf_path = os.path.join('results_distribution.pdf')
    plt.savefig(pdf_path, format='pdf')
    plt.close()




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

################################################################################
#                                 7 Main program                               #
################################################################################



def echo_management():
    """Main program loop"""
    total_penalty_learning_vector = []
    total_penalty_trivial_vector = []
    ############################################################################
    #                          8 Set up environment                            #
    ############################################################################

    # Set up game environment
    sim1 = echo1(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            render_env=render_env)
    sim2 = echo2(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            render_env=render_env)

    ############################################################################
    #                    9 Set up policy and target nets                       #
    ############################################################################

    # Set up policy and target neural nets
    policy_net = DQN(load_path='policy_net.pth')
    policy_net.eval()

    ############################################################################
    #                     11 Set up + start training loop                      #
    ############################################################################

    # Set up run counter and learning loop
    run = 0
    Continue = True



    # Continue repeating games (episodes) until target complete
    while Continue:

        ########################################################################
        #                           12 Play episode                            #
        ########################################################################

        # Increment run (episode) counter
        run += 1

        ########################################################################
        #                             13 Reset game                            #
        ########################################################################

        # Reset game environment and get first state observations
        random.seed(run+90000)
        np.random.seed(run+90000)
        state, info = sim1.reset(seed=run+90000, rate_sonographer_leave = rate_sonographer_leave )
        sim2.reset(seed=run+90000, rate_sonographer_leave = rate_sonographer_leave)


        # Reset step count for episode
        total_penalty_learning = 0
        total_penalty_trivial = 0
        terminated =False

        # Continue loop until episode complete
        while not terminated:

            ########################################################################
            #                       14 Game episode loop                           #
            ########################################################################


            ####################################################################
            #                       15 Get action                              #
            ####################################################################

            # Get action to take (se eval mode to avoid dropout layers)
            policy_net.eval()
            action = policy_net.act(state)

            ####################################################################
            #                 16 Play action (get S', R, T)                    #
            ####################################################################

            # Act
            state_next, reward, terminated, truncated, info = sim1.step(action)
            total_penalty_learning += (-1)*reward
            state = state_next

            state_next, action, reward, terminated, truncated, info = sim2.step(action_index = 2)
            total_penalty_trivial += (-1)*reward





            ####################################################################
            #                  18 Check for end of episode                     #
            ####################################################################

            # Actions to take if end of game episode
            if terminated:
                # Clear print row content
                clear_row = '\r' + ' ' * 79 + '\r'
                # print(clear_row, end='')
                print(f'Run: {run}, ', end='')
                print(f'Learning policy total penalty: {total_penalty_learning}, ', end=' ')
                print(f'Trivial policy total penalty: {total_penalty_trivial}, ', end=' ')
                print()
                total_penalty_trivial_vector.append(total_penalty_trivial)
                total_penalty_learning_vector.append(total_penalty_learning)


                ################################################################
                #             18b Check for end of learning                    #
                ################################################################

                if run == EPISODES:
                    Continue = False

                # End episode loop
                break

        # Target reached. Plot results
        if len(total_penalty_trivial_vector)%50 == 0:
            plot_results(total_penalty_trivial_vector, total_penalty_learning_vector)
            plot_difference_average(total_penalty_trivial_vector, total_penalty_learning_vector)

        ############################################################################
    #                      21 Learning complete                  #
    ############################################################################
    plot_results(total_penalty_trivial_vector, total_penalty_learning_vector)
    plot_difference_average(total_penalty_trivial_vector, total_penalty_learning_vector)
echo_management()
