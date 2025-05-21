import matplotlib.pyplot as plt
import os
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
render_env = True
policy_index = 0

sim = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
           num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
           num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
           rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
           render_env=render_env)

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

# Generate x-ticks for every 30 minutes and prepare counts for plotting
xticks = [sim.convert_to_time(step) for step in range(sim.convert_to_step(time_start), sim.convert_to_step(time_close), 30)]

for time_step in time_steps:
    time_label = sim.convert_to_time(time_step)
    time_labels.append(time_label)
    nonfetal_counts.append(nonfetal_dict[time_step])
    fetal_counts.append(fetal_dict[time_step])

fs=20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)

# Plotting Non-Fetal Patients
plt.bar(time_labels, nonfetal_counts, color='skyblue', width=8, label='Non-fetal')
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Number of Non-Fetal Patients', fontsize=fs)
# plt.title('Scheduled Non-Fetal Patients Over Time', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-7)
plt.xticks(ticks=xticks, rotation=45)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'nonfetal_schedule.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()

# Plotting Fetal Patients
plt.bar(time_labels, fetal_counts, color='red', width=8, label='Fetal')
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Number of Fetal Patients', fontsize=fs)
# plt.title('Scheduled Fetal Patients Over Time', fontsize=fs)
plt.tick_params(axis='both', which='both', labelsize=fs-7)
plt.xticks(ticks=xticks, rotation=45)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', 'fetal_schedule.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()
