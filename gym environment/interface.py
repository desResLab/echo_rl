import gymnasium as gym
import Memory
from agent_DQN import DQN as DQN
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import random
from echo_gym import echo_register

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up memomry
memory = Memory.Memory()
# Set up policy and target neural nets
policy_net = DQN()
target_net = DQN()
# Copy weights from policy_net to target
target_net.load_state_dict(policy_net.state_dict())
total_penalty_learning_vector = []
# Set target net to eval rather than training mode
# We do not train target net - ot is copied from policy net at intervals
target_net.eval()
############################################################################
#                     11 Set up + start training loop                      #
############################################################################

# Set up run counter and learning loop
TRAINING_EPISODES = 20000
run = 0
all_steps = 0
continue_learning = True

from gymnasium.envs.registration import register, registry

env_id = 'EchoEnv-v0'
echo_register()
if env_id in registry:
    del registry[env_id]
env = gym.make('EchoEnv-v0')

# Continue repeating games (episodes) until target complete
while continue_learning:

    ########################################################################
    #                           12 Play episode                            #
    ########################################################################

    # Increment run (episode) counter
    run += 1

    ########################################################################
    #                             13 Reset game                            #
    ########################################################################
    random.seed(run)
    np.random.seed(run)
    # Reset game environment and get first state observations
    state, info = env.reset(seed=run)  # Optionally pass a seed for reproducibility

    # Reset step count for episode
    total_penalty_learning = 0
    step = 0
    terminated = False
    # Continue loop until episode complete
    while not terminated:

        ########################################################################
        #                       14 Game episode loop                           #
        ########################################################################

        # Incrememnt step counts
        step += 1
        all_steps += 1

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
        state_next, reward, terminated, truncated, info = env.step(action)
        total_penalty_learning += (-1)*reward



        ####################################################################
        #                  17 Add S/A/R/S/T to memory                      #
        ####################################################################

        # Record state, action, reward, new state & terminal
        memory.remember(state, action, reward, state_next, terminated)

        # Update state
        state = state_next

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
            print()
            total_penalty_learning_vector.append(total_penalty_learning)


            ################################################################
            #             18b Check for end of learning                    #
            ################################################################

            if run == TRAINING_EPISODES:
                continue_learning = False

            # End episode loop
            break

        ####################################################################
        #                        19 Update policy net                      #
        ####################################################################

        # Avoid training model if memory is not of sufficient length
        if len(memory.memory) > (env.unwrapped.convert_to_step(env.unwrapped.time_close) + 120)*10 and all_steps%policy_net.Stride_size == 0:

            # Update policy net
            policy_net.optimize(target_net, memory.memory, policy_net.learning_rate)

            ################################################################
            #             20 Update target net periodically                #
            ################################################################

            # Use load_state_dict method to copy weights from policy net
            if all_steps % (env.unwrapped.convert_to_step(env.unwrapped.time_close) + 120) == 0:
                    target_net.load_state_dict(policy_net.state_dict())

    if len(total_penalty_learning_vector) > 2000:
        if policy_net.learning_rate > policy_net.LEARNING_RATE_MIN:
            policy_net.learning_rate *= policy_net.LEARNING_RATE_DECAY

############################################################################
#                      21 Learning complete                                #
############################################################################
def plot_penalty_average(total_penalty_learning_vector):
    # Convert to NumPy arrays for element-wise subtraction
    total_penalty_learning_vector = np.array(total_penalty_learning_vector)

    # Ensure vectors are the same length (optional step if necessary)
    min_len = min(len(total_penalty_learning_vector), len(total_penalty_learning_vector))
    total_penalty_learning_vector = total_penalty_learning_vector[:min_len]
    cumulative_sum = np.cumsum(total_penalty_learning_vector)

    # Compute the reward difference
    progressive_mean_diff = cumulative_sum / (np.arange(1, len(total_penalty_learning_vector) + 1))

    fs = 20
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text', usetex=True)
    # Create the plot
    # Create a range for the x-axis that starts from 1 and goes to the length of the vectors
    plt.xlabel('Time (days)', fontsize=fs)
    x_values = range(1, len(total_penalty_learning_vector) + 1)
    plt.figure(figsize=(10, 6))
    plt.xticks([TRAINING_EPISODES])
    # Plot the average reward difference curve
    plt.plot(x_values, progressive_mean_diff, label="Penalty Average", color='blue')
    # Add labels and title
    plt.xlabel("Time (days)", fontsize=fs)
    plt.ylabel("Penalty Average", fontsize=fs)
    plt.title("Average of the Penalty wrt Time", fontsize=fs)


    # Add a legend to differentiate the two curves
    plt.legend(fontsize=fs-7)

    # Display the plot
    plt.tight_layout()
    plt.grid(True)
    pdf_path = os.path.join('penalty_average.pdf')
    plt.savefig(pdf_path, format='pdf')
    plt.close()
plot_penalty_average(total_penalty_learning_vector)
model_save_path = os.path.join('policy_net.pth')
torch.save(policy_net.state_dict(), model_save_path)
print(f'Model parameters saved to {model_save_path}')

