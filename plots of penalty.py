import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from simulation import time_close, render_env, sim, penalty
import random
for policy in range(1,9):
    for iteration in range(365):
        random.seed(iteration)
        sim.reset()  # Reset simulation for each policy
        current_time = sim.convert_to_step('08:00')  # Convert the time directly to the desired format
        end_time = sim.convert_to_step(time_close) + 120
        while current_time <= end_time:
            sim.step(policy, render_env)
            penalty[policy][current_time] += sim._penalty_calculation()/365
            sim.env.run(until=current_time + 1)
            current_time += 1

start_time = datetime.strptime('08:00', '%H:%M')
time_range = [start_time + timedelta(minutes=i) for i in range(len(penalty[1]))]  # Adjust based on number of policies

# Plotting
plt.figure(figsize=(12, 6))
for i in range(len(penalty)):
    if i != 0:
        plt.plot(time_range, penalty[i], label=f'Policy {i}')

plt.xlabel('Time')
plt.ylabel('Penalty')
plt.title('Penalty for Each Policy')
plt.legend()
plt.grid(True)
# Set x-axis ticks to represent time in "00:00" format
plt.xticks(time_range[::60], [time.strftime('%H:%M') for time in time_range[::60]], rotation=45)
plt.tight_layout()
plt.show()
