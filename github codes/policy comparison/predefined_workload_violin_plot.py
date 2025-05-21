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
    results = {p: [] for p in policies}
    num_sonographers = num_sonographer_both + num_sonographer_nonfetal

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

            # List to store daily completed patients
            daily_completed = []

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
                    sim.step(policy)
                    current_time += 1

                # Count completed patients
                completed = sum(1 for p in sim.state['patients'] if p['status'] == 'done')
                patients_per_sonographer = completed / num_sonographers
                daily_completed.append(patients_per_sonographer)

                # Print results for every day
                print(f"    Day {day}: Completed {completed} patients, {patients_per_sonographer:.2f} per sonographer")

            # Store this case's results
            results[policy].append(daily_completed)

    return results, policy_cases


def create_sonographer_workload_violin_plot(results, policy_cases):
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
                    data.append({"Policy": label, "Patients Per Sonographer": val})
        else:
            label = f"Policy {policy}"
            label_to_policy[label] = policy
            for val in policy_results[0]:
                data.append({"Policy": label, "Patients Per Sonographer": val})

    df = pd.DataFrame(data)

    # Assign a color for each policy (not each label)
    policies = sorted(set(label_to_policy.values()))
    base_palette = sns.color_palette("Set2", len(policies))
    policy_color_map = dict(zip(policies, base_palette))

    # Map each label to its policy color
    label_palette = {
        label: policy_color_map[label_to_policy[label]] for label in df["Policy"].unique()
    }

    # Plot
    plt.figure(figsize=(18, 10))
    sns.violinplot(
        x="Policy",
        y="Patients Per Sonographer",
        data=df,
        inner="quartile",
        scale="width",
        palette=label_palette,
#        cut=0  # Prevent KDE from spilling below min
    )

    fs = 25
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    try:
        plt.rc('text', usetex=True)  # Only use if LaTeX is available
    except:
        pass

    plt.xlabel("Policies", fontsize=fs)
    plt.ylabel("Patients Per Sonographer", fontsize=fs)
    plt.yticks(ha='right', fontsize=fs - 2)
    plt.xticks(rotation=45, ha='right', fontsize=fs - 2)
    plt.tight_layout()

    plt.savefig('sonographer_workload_violin.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig('sonographer_workload_violin.png', format='png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("Starting Echo Sonographer Workload Analysis...")

    # Run simulations and get results
    results, policy_cases = run_policy_simulations()

    # Create violin plot
    create_sonographer_workload_violin_plot(results, policy_cases)

    print("Analysis complete. Results saved as 'sonographer_workload_violin.pdf/png'")


if __name__ == "__main__":
    main()