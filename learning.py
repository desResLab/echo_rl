from Echo_class import Echo as echo1
from Echo_trivial import Echo as echo2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Discount rate of future rewards
GAMMA = 0.95
# Learing rate for neural network
LEARNING_RATE = 0.001
# Maximum number of game steps (state, action, reward, next state) to keep
MEMORY_SIZE = 1000000
# Sample batch size for policy network update
BATCH_SIZE = 10
# Exploration rate (epislon) is probability of choosing a random action
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.999
# Training episodes
TRAINING_EPISODES = 50


class DQN(nn.Module):
    """Deep Q Network. Used for both policy (action) and target (Q) networks."""

    def __init__(self):
        """Constructor method. Set up neural nets."""

        # Set starting exploration rate
        self.exploration_rate = EXPLORATION_MAX

        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 144)
        )

        # Initialize weights and biases to 0
        self.net.apply(self._init_weights)

        # Set loss function and optimizer
        self.objective = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=LEARNING_RATE)

    def _init_weights(self, m):
        """Initialize weights and biases to 0."""
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0)  # Set weights to 0
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
                                       state[3] + 1, 3)):
                        for m in range(min(state[6] + state[7] + 1 - i - j - k - l,
                                           state[8] + state[9] + 1 - i - j - k - l,
                                           state[4] + 1, 3)):
                            for n in range(min(state[6] + state[7] + 1 - i - j - k - l - m,
                                           state[8] + state[9] + 1 - i - j - k - l - m,
                                           state[5] + 1, 2)):
                                action.append([i, j, k, l, m, n])

        # Convert valid actions to integer representations
        for a in action:
            action_int.append(a[0] * 72 + a[1] * 36 + a[2] * 18 + a[3] * 6 + a[4] * 2 + a[5])

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

def optimize(policy_net, target_net, memory):
    alpha = 0.03

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
            expected_state_action_values[action] += alpha*(updated_q - expected_state_action_values[action])
        else:
            # For termal actions Q = reward (-1)
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach values from gradients to prevent gradient update
            expected_state_action_values = expected_state_action_values.detach()
            # Set Q for all actions to reward (-1)
            expected_state_action_values[:] = reward

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


################################################################################
#                            5 Define memory class                             #
################################################################################

class Memory():
    """
    Replay memory used to train model.
    Limited length memory (using deque, double ended queue from collections).
      - When memory full deque replaces oldest data with newest.
    Holds, state, action, reward, next state, and episode done.
    """

    def __init__(self):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=MEMORY_SIZE)

    def remember(self, state, action, reward, next_state, done):
        """state/action/reward/next_state/done"""
        self.memory.append((state, action, reward, next_state, done))


def plot_results(total_reward_trivial_vector, total_reward_learning_vector):
    # Create a range for the x-axis that starts from 1 and goes to the length of the vectors
    x_values = range(1, len(total_reward_trivial_vector) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the trivial reward curve
    plt.plot(x_values, total_reward_trivial_vector, label="Trivial Policy", color='blue', marker='o')

    # Plot the learning reward curve
    plt.plot(x_values, total_reward_learning_vector, label="Learning Policy", color='green', marker='x')

    # Add labels and title
    plt.xlabel("Index")
    plt.ylabel("Total Reward")
    plt.title("Total Reward Comparison: Trivial vs Learning Policy")

    # Ensure that x-ticks are integers
    plt.xticks(x_values)

    # Add a legend to differentiate the two curves
    plt.legend()

    # Display the plot
    plt.grid(True)
    plt.show()



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
    policy_net = DQN()
    target_net = DQN()

    # Copy weights from policy_net to target
    target_net.load_state_dict(policy_net.state_dict())

    # Set target net to eval rather than training mode
    # We do not train target net - ot is copied from policy net at intervals
    target_net.eval()

    ############################################################################
    #                            10 Set up memory                              #
    ############################################################################

    # Set up memomry
    memory = Memory()

    ############################################################################
    #                     11 Set up + start training loop                      #
    ############################################################################

    # Set up run counter and learning loop
    run = 0
    all_steps = 0
    continue_learning = True



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

        # Reset game environment and get first state observations
        random.seed(run)
        np.random.seed(run)
        state = sim1.reset()
        sim1._convert_early_patients()
        sim2.reset()


        # Reset step count for episode
        total_penalty_learning = 0
        total_penalty_trivial = 0
        step = 0

        # Continue loop until episode complete
        while step <= sim1.convert_to_step(time_close) + 120:

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
            reward, terminal = sim1.step(action, False)
            sim2.step(1, False)
            total_penalty_learning += (-1)*reward
            total_penalty_trivial += sim2._penalty_calculation()


            # Advance simulation ready for next action
            if not terminal:
                sim1.env.run(until=step + 1)
                sim1._convert_early_patients()
                sim2.env.run(until=step + 1)

            # Reshape state into 2D array with state obsverations as first 'row'
            state_next = [sim1._ontime_waiting_fetal_patients(), sim1._late_waiting_fetal_patients(), sim1._early_waiting_fetal_patients(),
                 sim1._ontime_waiting_nonfetal_patients(), sim1._late_waiting_nonfetal_patients(), sim1._early_waiting_nonfetal_patients(),
                 sim1._available_fetal_sonograpphers(), sim1._available_nonfetal_sonograpphers(), sim1._available_fetal_rooms(),
                 sim1._available_nonfetal_rooms(), sim1._onbreak_fetal_sonograpphers(), sim1._onbreak_nonfetal_sonograpphers(),
                 sim1.env.now]


            ####################################################################
            #                  17 Add S/A/R/S/T to memory                      #
            ####################################################################

            # Record state, action, reward, new state & terminal
            memory.remember(state, action, reward, state_next, terminal)

            # Update state
            state = state_next

            ####################################################################
            #                  18 Check for end of episode                     #
            ####################################################################

            # Actions to take if end of game episode
            if terminal:
                # Clear print row content
                clear_row = '\r' + ' ' * 79 + '\r'
                print(clear_row, end='')
                print(f'Run: {run}, ', end='')
                print(f'Learning policy total penalty: {total_penalty_learning}, ', end=' ')
                print(f'Trivial policy total penalty: {total_penalty_trivial}, ', end='')
                total_penalty_learning_vector.append(total_penalty_learning)
                total_penalty_trivial_vector.append(total_penalty_trivial)


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
            if len(memory.memory) > (sim1.convert_to_step(time_close) + 120)*10:

                # Update policy net
                optimize(policy_net, target_net, memory.memory)

                ################################################################
                #             20 Update target net periodically                #
                ################################################################

                # Use load_state_dict method to copy weights from policy net
                if all_steps % (sim1.convert_to_step(time_close) + 120) == 0:
                    target_net.load_state_dict(policy_net.state_dict())

    ############################################################################
    #                      21 Learning complete - plot results                 #
    ############################################################################

    # Target reached. Plot results
    plot_results(total_penalty_trivial_vector, total_penalty_learning_vector)
echo_management()