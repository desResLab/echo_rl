import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from simulation import time_close, render_env, sim, penalty
import numpy as np
import random
import os
for iteration in range(365):
    for policy in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]:
        random.seed(iteration)
        np.random.seed(iteration)
        sim.reset()  # Reset simulation for each policy
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
        current_time = sim.convert_to_step('08:00')  # Convert the time directly to the desired format
        end_time = sim.convert_to_step(time_close) + 120

        while current_time <= end_time:
            if policy < 5:
                sim.step(policy, render_env)
            else:
                sim.step(5, render_env)
            penalty[policy][current_time] += sim._penalty_calculation() / 365
            sim.env.run(until=current_time + 1)
            current_time += 1


start_time = datetime.strptime('08:00', '%H:%M')
time_range = [start_time + timedelta(minutes=i) for i in range(len(penalty[1]))]  # Adjust based on number of policies

# Plotting
plt.figure(figsize=(12, 6))
# Define a list of colors
colors = [
    'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow',
    'black', 'pink', 'brown', 'gray', 'olive', 'lightblue'
]

for i in range(1, 15):
    if i < 5:
        plt.plot(time_range, penalty[i], label=f'Policy {i}', color=colors[i-1])
    elif i == 5:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0)', color=colors[i-1])
    elif i == 6:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0.25)', color=colors[i-1])
    elif i == 7:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0.5)', color=colors[i-1])
    elif i == 8:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=0.75)', color=colors[i-1])
    elif i == 9:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=0, beta=1)', color=colors[i-1])
    elif i == 10:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0)', color=colors[i-1])
    elif i == 11:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0.25)', color=colors[i-1])
    elif i == 12:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0.5)', color=colors[i-1])
    elif i == 13:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=0.75)', color=colors[i-1])
    elif i == 14:
        plt.plot(time_range, penalty[i], label='Policy 5 (alpha=1, beta=1)', color=colors[i-1])

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
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'penalty plot 4.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()
