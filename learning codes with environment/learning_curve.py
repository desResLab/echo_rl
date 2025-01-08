import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def plot_combined_metrics(penalty_vector, leave_rate_vector, output_path="combined_plot.pdf"):
    """
    Creates a single plot showing both the smoothed logarithm of total penalty
    and sonographer leave rate over time on the same axes.

    Parameters:
    penalty_vector (np.array): Vector containing total penalty values
    leave_rate_vector (np.array): Vector containing sonographer leave rates
    output_path (str): Path where the PDF file will be saved
    """
    # Setup the plot style
    fs = 20
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')
    plt.rc('text', usetex=True)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Process penalty data
    log_penalty = np.log1p(np.array(penalty_vector))
    smoothed_penalty = (
        pd.Series(log_penalty)
        .rolling(window=50, min_periods=1)
        .mean()
        .to_list()
    )

    # Create x values and specify ticks
    x_values = range(1, len(penalty_vector) + 1)
    xticks = [1, 10000, 20000, 30000]
    xticks = [tick for tick in xticks if tick <= len(penalty_vector)]

    # Plot penalty data on first y-axis
    line1 = ax1.plot(x_values, smoothed_penalty, 'b-', label="Smoothed Log Penalty")
    ax1.set_xlabel("Time (days)", fontsize=fs)
    ax1.set_ylabel("Logarithm of Total Penalty", fontsize=fs, color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Create second y-axis for leave rate
    ax2 = ax1.twinx()
    line2 = ax2.plot(x_values[:len(leave_rate_vector)], leave_rate_vector, 'r-',
                     label="Sonographer Leave Rate")
    ax2.set_ylabel("Sonographer Leave Rate", fontsize=fs, color='r', rotation=270, labelpad=20)

    # Set y-ticks from 0.1 to 1.0
    yticks = np.linspace(0.1, 1.0, 10)  # Creates 10 evenly spaced ticks from 0.1 to 1.0
    ax2.set_yticks(yticks)
    ax2.set_yticklabels([f'{x:.1f}' for x in yticks])
    ax2.tick_params(axis='y', labelcolor='r', labelsize=fs)

    # Set x-axis ticks
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([str(tick) for tick in xticks])

    # Add title and adjust tick sizes
    plt.title("Combined Metrics over Time", fontsize=fs)
    ax1.tick_params(axis='both', which='both', labelsize=fs - 2)
    ax2.tick_params(axis='both', which='both', labelsize=fs - 2)

    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=fs - 2, loc='upper right')

    # Add grid
    ax1.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()


# Load the data
penalty_path = "penalty_vectors/total_penalty_learning_vector_final.npy"
leave_rate_path = "sonographer_vectors/rate_sonographer_leave_vector_final.npy"

if os.path.exists(penalty_path) and os.path.exists(leave_rate_path):
    penalty_vector = np.load(penalty_path)
    leave_rate_vector = np.load(leave_rate_path)
    plot_combined_metrics(penalty_vector, leave_rate_vector)
else:
    print("One or both input files not found!")