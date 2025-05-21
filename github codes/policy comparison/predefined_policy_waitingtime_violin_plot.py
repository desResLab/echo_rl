import random
import numpy as np
from Echo_trivial import Echo
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def run_policy_simulations():
    # Basic simulation parameters
    time_start = '08:00'
    time_close = '17:00'
    num_fetal_room = 1
    num_nonfetal_room = 6
    num_sonographer_both = 4
    num_sonographer_nonfetal = 2
    time_sonographer_break = 15
    rate_sonographer_leave = 0.1
    rate_absence = 0.1
    render_env = False

    # Define the policies and cases to evaluate
    policies = [1, 2, 3, 4, 5, 6]

    # Alpha and beta values for policies 5 and 6
    # Alpha only takes 0 or 1, beta takes values from 0 to 1
    policy_cases = {
        5: [
            {'alpha': 0, 'beta': 0.0},  # Case 1
            {'alpha': 0, 'beta': 0.25},  # Case 2
            {'alpha': 0, 'beta': 0.5},  # Case 3
            {'alpha': 0, 'beta': 0.75},  # Case 4
            {'alpha': 0, 'beta': 1.0},  # Case 5
            {'alpha': 1, 'beta': 0.0},  # Case 6
            {'alpha': 1, 'beta': 0.25},  # Case 7
            {'alpha': 1, 'beta': 0.5},  # Case 8
            {'alpha': 1, 'beta': 0.75},  # Case 9
            {'alpha': 1, 'beta': 1.0}  # Case 10
        ],
        6: [
            {'alpha': 0, 'beta': 0.0},  # Case 1
            {'alpha': 0, 'beta': 0.25},  # Case 2
            {'alpha': 0, 'beta': 0.5},  # Case 3
            {'alpha': 0, 'beta': 0.75},  # Case 4
            {'alpha': 0, 'beta': 1.0},  # Case 5
            {'alpha': 1, 'beta': 0.0},  # Case 6
            {'alpha': 1, 'beta': 0.25},  # Case 7
            {'alpha': 1, 'beta': 0.5},  # Case 8
            {'alpha': 1, 'beta': 0.75},  # Case 9
            {'alpha': 1, 'beta': 1.0}  # Case 10
        ]
    }

    # For policies 1-4, add a single dummy case
    for policy in [1, 2, 3, 4]:
        policy_cases[policy] = [{'alpha': 0, 'beta': 0.0}]

    # Results storage
    results = {}
    for policy in policies:
        results[policy] = []

    # Get time steps
    sim_ref = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
                   num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
                   num_sonographer_nonfetal=num_sonographer_nonfetal,
                   time_sonographer_break=time_sonographer_break,
                   rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
                   render_env=render_env)
    closing_time_step = sim_ref.convert_to_step(time_close)

    # Number of days to simulate
    num_days = 365  # Set to 365 for final results

    # Run simulations for each policy
    for policy in policies:
        print(f"Processing Policy {policy}")

        # For each case of the current policy
        for case_idx, case_params in enumerate(policy_cases[policy]):
            alpha_val = case_params['alpha']
            beta_val = case_params['beta']
            print(f"  Case {case_idx + 1}: alpha={alpha_val}, beta={int(beta_val * 100)}%")

            # List to store daily waiting times
            case_waiting_times = []

            # Simulate for multiple days
            for day in range(num_days):
                # Initialize simulation
                sim = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
                           num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
                           num_sonographer_nonfetal=num_sonographer_nonfetal,
                           time_sonographer_break=time_sonographer_break,
                           rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
                           render_env=render_env)

                # Set seeds
                random.seed(day)
                np.random.seed(day)

                # Reset simulation
                sim.reset(rate_sonographer_leave=rate_sonographer_leave)

                # For policies 5 and 6, use _make_reservations
                if policy in [5, 6]:
                    sim._make_reservations(alpha_val, beta_val)

                # Run simulation
                current_time = 0
                while current_time <= closing_time_step + 120:
                    # Run step
                    sim.step(policy)
                    current_time += 1

                # At the end of simulation, collect all waiting times, treating negative values as 0
                waiting_times = []
                for patient in sim.state['patients']:
                    # Check if patient has arrived (has a waiting time value)
                    if 'waiting time' in patient and patient['waiting time'] != 'NA' and isinstance(
                            patient['waiting time'], (int, float)):
                        # Record negative waiting times as 0, keep positive values as they are
                        wait_time = max(0, patient['waiting time'])
                        waiting_times.append(wait_time)

                # Calculate average waiting time
                if waiting_times:
                    avg_waiting_time = sum(waiting_times) / len(waiting_times)
                else:
                    avg_waiting_time = 0

                # Store this day's result
                case_waiting_times.append(avg_waiting_time)

                # Print status
                completed = sum(1 for p in sim.state['patients'] if p['status'] == 'done')
                total = len(sim.state['patients'])
                waiting = sum(1 for p in sim.state['patients'] if 'waiting' in p['status'])

                print(f"    Day {day}: Completed {completed}/{total}, "
                      f"Waiting times: {len(waiting_times)}, "
                      f"Still waiting: {waiting}, "
                      f"Avg wait: {avg_waiting_time:.2f} min")

            # Store this case's results
            results[policy].append(case_waiting_times)

    return results, policy_cases


def create_waiting_time_violin_plot(results, policy_cases):

    # Prepare data
    data = []
    label_to_policy = {}
    for policy in sorted(results.keys()):
        policy_results = results[policy]
        if len(policy_results) > 1:
            for case_idx, daily_values in enumerate(policy_results):
                alpha_val = policy_cases[policy][case_idx]['alpha']
                beta_pct = int(policy_cases[policy][case_idx]['beta'] * 100)
                label = f"Policy {policy} ($\\alpha$={alpha_val*100:.0f}%, $\\beta$={beta_pct}%)"
                label_to_policy[label] = policy
                for val in daily_values:
                    data.append({"Policy": label, "Waiting Time": max(0, val)})
        else:
            label = f"Policy {policy}"
            label_to_policy[label] = policy
            for val in policy_results[0]:
                data.append({"Policy": label, "Waiting Time": max(0, val)})

    df = pd.DataFrame(data)

    # Remove any negative waiting times (just in case)
    df = df[df["Waiting Time"] >= 0]

    # Assign a color for each policy (not each label)
    policies = sorted(set(label_to_policy.values()))
    base_palette = sns.color_palette("Set2", len(policies))
    policy_color_map = dict(zip(policies, base_palette))

    # Map each label (e.g., P5\nα=1\nβ=50%) to its policy color
    label_palette = {
        label: policy_color_map[label_to_policy[label]] for label in df["Policy"].unique()
    }

    # Plot
    plt.figure(figsize=(18, 10))
    sns.violinplot(
        x="Policy",
        y="Waiting Time",
        data=df,
        inner="quartile",
        scale="width",
        palette=label_palette,
        cut=0  # Prevent KDE from spilling below min
    )
    fs = 25
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text', usetex=True)

    plt.xlabel("Policies", fontsize=fs)
    plt.ylabel("Waiting Time (minutes)", fontsize=fs)
    plt.yticks(ha='right', fontsize=fs - 2)
    plt.xticks(rotation=45, ha='right', fontsize=fs-2)
    plt.tight_layout()

    plt.savefig('waiting_time_by_policy_violin.pdf', format='pdf', dpi=300)
    plt.savefig('waiting_time_by_policy_violin.png', format='png', dpi=300)
    plt.close()


def main():
    print("Starting Echo Policy Analysis...")

    # Run simulations and get results
    results, policy_cases = run_policy_simulations()

    # Create bar plot
    create_waiting_time_violin_plot(results, policy_cases)

    print("Analysis complete. Results saved as 'waiting_time_by_policy_CI_withRL.pdf' and 'waiting_time_by_policy_CI.png'")


if __name__ == "__main__":
    main()