import random
import numpy as np
from Echo_trivial_ab import Echo
from Echo_class_ab import Echo as Echo1
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the DQN model (unchanged from original)
num_action = 3


class DQN(nn.Module):
    # [DQN implementation unchanged from original]
    def __init__(self, load_path=None):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, num_action ** 6)
        )
        # Load the model parameters if a path is provided
        if load_path:
            self.load_model(load_path)

    def load_model(self, path):
        """Load pre-trained model parameters from the given path, adjusting keys if necessary."""
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

            # Remove the "net." prefix from keys if present
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("net."):
                    new_key = key.replace("net.", "")
                else:
                    new_key = key
                new_state_dict[new_key] = value

            self.net.load_state_dict(new_state_dict)
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Initializing with random weights")

    def act(self, state):
        """Select action based on current state using the policy network."""

        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        action = []
        action_int = []

        for i in range(min(state[0] + 1, num_action)):
            for j in range(min(state[1] + 1, num_action)):
                for k in range(min(state[2] + 1, num_action)):
                    for l in range(min(state[3] + 1, num_action)):
                        for m in range(min(state[4] + 1, num_action)):
                            for n in range(min(state[5] + 1, num_action)):
                                action.append([i, j, k, l, m, n])

        # Convert valid actions to integer representations
        for a in action:
            action_int.append(
                a[0] * num_action ** 5 + a[1] * num_action ** 4 + a[2] * num_action ** 3 +
                a[3] * num_action ** 2 + a[4] * num_action ** 1 + a[5])

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


def run_policy_simulations():
    # Basic simulation parameters
    time_start = '08:00'
    time_close = '17:00'
    rate_sonographer_leave = 0.1
    rate_absence = 0.1
    render_env = False

    # Set the seed for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Define the policies to evaluate: Policies 1, 2, 3, 4 and RL
    policies = [0, 1, 2, 3, 4]  # Policy 0 is the RL policy

    # Results storage - Changed to store daily averages
    results = {policy: [] for policy in policies}

    # Get time steps from Echo (non-RL) for reference
    sim_ref = Echo(time_start=time_start, time_close=time_close, rate_absence=rate_absence,
                   render_env=render_env)
    closing_time_step = sim_ref.convert_to_step(time_close)

    # Initialize the DQN policy network
    policy_net = DQN(load_path='models/policy_net_final_ab.pth')
    policy_net.eval()  # Set to evaluation mode

    # Number of days to simulate
    num_days = 365  # Set to 365 for final results

    # Run simulations for each policy
    for policy in policies:
        policy_name = "RL" if policy == 0 else f"P{policy}"
        print(f"Processing Policy {policy_name}")

        # Simulate for multiple days
        for day in range(num_days):
            # Set seeds for consistency across policies
            random.seed(day)
            np.random.seed(day)
            torch.manual_seed(day)

            # Initialize simulation - using Echo1 for RL policy, Echo for others
            if policy == 0:
                # Use Echo1 for the RL policy
                sim = Echo1(time_start=time_start, time_close=time_close,  rate_absence=rate_absence,
                            render_env=render_env)
                # Reset the RL environment
                state, info = sim.reset(seed=day)
            else:
                # Use Echo for standard policies
                sim = Echo(time_start=time_start, time_close=time_close, rate_absence=rate_absence,
                           render_env=render_env)
                # Reset standard environment
                sim.reset(seed=day, rate_sonographer_leave=rate_sonographer_leave)

            # Run simulation
            current_time = 0
            terminated = False

            while current_time <= closing_time_step + 120 and not terminated:
                # Run step with appropriate policy
                if policy == 0:
                    # Use the RL policy (policy 0) with Echo1
                    action = policy_net.act(state)
                    state_next, reward, terminated, truncated, info = sim.step(action)
                    state = state_next
                else:
                    # Use existing policies 1, 2, 3, or 4 with Echo
                    sim.step(policy)
                    terminated = False  # Standard policies don't have termination

                current_time += 1

            # At the end of simulation, collect waiting times and calculate average
            waiting_times = []
            for patient in sim.state['patients']:
                # Check if patient has arrived (has a waiting time value)
                if 'waiting time' in patient and patient['waiting time'] != 'NA' and isinstance(patient['waiting time'],
                                                                                                (int, float)):
                    # Record negative waiting times as 0, keep positive values as they are
                    wait_time = max(0, patient['waiting time'])
                    waiting_times.append(wait_time)
                # Skip patients who haven't arrived (no waiting time value)

            # Calculate average waiting time for this day
            if waiting_times:
                avg_waiting_time = sum(waiting_times) / len(waiting_times)
            else:
                avg_waiting_time = 0

            # Store the day's average waiting time
            results[policy].append(avg_waiting_time)

            # Print status
            completed = sum(1 for p in sim.state['patients'] if p['status'] == 'done')
            total = len(sim.state['patients'])
            waiting = sum(1 for p in sim.state['patients'] if 'waiting' in p['status'])

            print(f"    Day {day}: Completed {completed}/{total}, "
                  f"Still waiting: {waiting}, "
                  f"Avg wait: {avg_waiting_time:.2f} min")

    return results


def create_waiting_time_violin_plot(results):
    """
    Create a violin plot of daily average waiting times.

    Parameters:
    - results: Dictionary of simulation results (policy -> list of daily avg wait times)
    """
    # Prepare data for plotting
    data = []

    for policy in sorted(results.keys()):
        # Get the policy name for display
        if policy == 0:
            policy_name = "RL policy"
        else:
            policy_name = f"Policy {policy}"

        # Add each day's average waiting time as a separate row in the dataframe
        for avg_wait_time in results[policy]:
            data.append({"Policy": policy_name, "Average Waiting Time": avg_wait_time})

    # Convert to pandas DataFrame for seaborn
    df = pd.DataFrame(data)

    # Remove any negative waiting times (just in case)
    df = df[df["Average Waiting Time"] >= 0]

    # Create color palette
    policies = sorted(df["Policy"].unique())
    color_palette = sns.color_palette("Set2", len(policies))
    policy_colors = dict(zip(policies, color_palette))

    fs = 20
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text', usetex=True)

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Create violin plot
    sns.violinplot(
        x="Policy",
        y="Average Waiting Time",
        data=df,
        inner="quartile",  # Shows quartiles inside violin
        scale="width",  # Makes all violins the same width
        palette=policy_colors,
        cut=0  # Prevent KDE from spilling below min
    )

    # Formatting
    plt.xlabel("Policies", fontsize=fs)
    plt.ylabel("Average Waiting Time (minutes)", fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save the figure
    plt.tight_layout()
    plt.savefig('daily_avg_waiting_time_by_policy_violin_ab.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Starting Echo Policy Analysis with RL...")

    # Run simulations and get results
    results = run_policy_simulations()

    # Create violin plot
    create_waiting_time_violin_plot(results)

    print(
        "Analysis complete. Results saved as 'daily_avg_waiting_time_by_policy_violin.pdf' and 'daily_avg_waiting_time_by_policy_violin.png'")


if __name__ == "__main__":
    main()