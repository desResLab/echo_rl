import matplotlib.pyplot as plt
from Echo_class import Echo

# Initialize the Echo class with imported parameters
time_start = '08:00'
time_close = '17:00'
num_fetal_room = 1
num_nonfetal_room = 6
num_sonographer_both = 4
num_sonographer_nonfetal = 2
time_sonographer_break = 15
rate_sonographer_leave = 0.1
rate_absence = 0.1
ontime_anger = 2
late_anger = 0
render_env = True
policy_index = 0

sim = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
           num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
           num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
           rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
           ontime_anger=ontime_anger, late_anger=late_anger, render_env=render_env)

sim._load_schedules()

# Extract time steps and patients
time_steps = range(sim.convert_to_step(sim.time_start), sim.convert_to_step(sim.time_close))
patients = sim.state['patients']

# Prepare data for plotting
time_labels = []
nonfetal_counts = []
fetal_counts = []

# Create dictionaries to count the number of fetal and non-fetal patients at each time step
nonfetal_dict = {time: 0 for time in time_steps}
fetal_dict = {time: 0 for time in time_steps}

# Count patients at each time step
for patient in patients:
    schedule_time = patient['Schedule time']
    if patient['fetal'] == 'no':
        nonfetal_dict[schedule_time] += 1
    elif patient['fetal'] == 'yes':
        fetal_dict[schedule_time] += 1

# Convert time steps to time labels and prepare counts for plotting
for time_step in time_steps:
    nonfetal_count = nonfetal_dict[time_step]
    fetal_count = fetal_dict[time_step]

    if nonfetal_count > 0 or fetal_count > 0:
        time_labels.append(sim.convert_to_time(time_step))
        nonfetal_counts.append(nonfetal_count)
        fetal_counts.append(fetal_count)

# Plotting Non-Fetal Patients
plt.figure(figsize=(15, 6))
plt.bar(time_labels, nonfetal_counts, color='skyblue', width=0.4, label='Non-fetal')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Number of Non-Fetal Patients', fontsize=14)
plt.title('Scheduled Non-Fetal Patients Over Time', fontsize=18)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting Fetal Patients
plt.figure(figsize=(15, 6))
plt.bar(time_labels, fetal_counts, color='red', width=0.4, label='Fetal')
plt.xlabel('Time', fontsize=14)
plt.ylabel('Number of Fetal Patients', fontsize=14)
plt.title('Scheduled Fetal Patients Over Time', fontsize=18)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

