from Echo_class import Echo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import numpy as np
import statistics
import random
import torch.nn as nn
import torch

# parameters
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
# Set the seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
EPISODES = 10000
class DQN(nn.Module):
    def __init__(self, load_path):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(96, 64)
        )
        # Load the model parameters if a path is provided
        self.load_model('models/policy_net_final.pth')



    def load_model(self, path):
        """Load pre-trained model parameters from the given path, adjusting keys if necessary."""
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        # Remove the "net." prefix from keys
        new_state_dict = {key.replace("net.", ""): value for key, value in state_dict.items()}

        self.net.load_state_dict(new_state_dict)
        print(f"Model loaded from {path}")

    def act(self, state):
        """Select action based on current state using the policy network."""

        # Convert state to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        action = []
        action_int = []
        fetal_pair = min(state[6], state[8])

        # Generate all possible valid actions
        for i in range(min(fetal_pair + 1, state[0] + 1, 2)):
            for j in range(min(fetal_pair + 1 - i, state[1] + 1, 2)):
                for k in range(min(fetal_pair + 1 - i - j, state[2] + 1, 2)):
                    for l in range(min(state[6] + state[7] + 1 - i - j - k,
                                        state[8] + state[9] + 1 - i - j - k,
                                        state[3] + 1, 2)):
                        for m in range(min(state[6] + state[7] + 1 - i - j - k - l,
                                            state[8] + state[9] + 1 - i - j - k - l,
                                            state[4] + 1, 2)):
                            for n in range(min(state[6] + state[7] + 1 - i - j - k - l - m,
                                                state[8] + state[9] + 1 - i - j - k - l - m,
                                                state[5] + 1, 2)):
                                action.append([i, j, k, l, m, n])

        # Convert valid actions to integer representations
        for a in action:
            action_int.append(a[0] * 32 + a[1] * 16 + a[2] * 8 + a[3] * 4 + a[4] * 2 + a[5])

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

policy_net = DQN(load_path='policy_net.pth')
policy_net.eval()
sim = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            render_env=render_env)
rows = sim.convert_to_step(time_close) +121
ontime_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
late_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
early_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
fetal_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
nonfetal_waiting_patients = [[0 for _ in range(rows)] for _ in range(365)]
available_fetal_room = [0 for _ in range(rows)]
available_nonfetal_room = [0 for _ in range(rows)]
available_fetal_sonographer = [0 for _ in range(rows)]
available_nonfetal_sonographer = [0 for _ in range(rows)]
current_time = sim.convert_to_step(time_start)
for iteration in range(365):
    current_time = 0  # Reset current time for each policy
    random.seed(iteration)
    np.random.seed(iteration)
    state, info = sim.reset(seed= iteration, rate_sonographer_leave = 0.1)
    ontime_waiting_patients[iteration][current_time] = sim._ontime_waiting_patients()
    late_waiting_patients[iteration][current_time] = sim._late_waiting_patients()
    early_waiting_patients[iteration][current_time] = sim._early_waiting_patients()
    fetal_waiting_patients[iteration][current_time] = sim._fetal_waiting_patients()
    nonfetal_waiting_patients[iteration][current_time] = sim._nonfetal_waiting_patients()
    available_fetal_room[current_time] += (len([room for room in sim.state['rooms'] if
                                                room['fetal'] in ['yes', 'yes (reserved)'] and room[
                                                    'status'] == 'available'])) / 365
    available_nonfetal_room[current_time] += (
                                                 len([room for room in sim.state['rooms'] if
                                                      room['fetal'] == 'no' and room['status'] == 'available'])) / 365
    available_fetal_sonographer[current_time] += (
                                                     len([sonographer for sonographer in sim.state['sonographers'] if
                                                          sonographer['fetal'] in ['yes', 'yes (reserved)'] and
                                                          sonographer['status'] == 'available'])) / 365
    available_nonfetal_sonographer[current_time] += (
                                                        len([sonographer for sonographer in sim.state['sonographers'] if
                                                             sonographer['fetal'] == 'no' and sonographer[
                                                                 'status'] == 'available'])) / 365
    while current_time <= sim.convert_to_step(time_close) + 120 :
        action = policy_net.act(state)
        state_next, reward, terminated, truncated, info = sim.step(action)
        state = state_next
        ontime_waiting_patients[iteration][current_time] = sim._ontime_waiting_patients()
        late_waiting_patients[iteration][current_time] = sim._late_waiting_patients()
        early_waiting_patients[iteration][current_time] = sim._early_waiting_patients()
        fetal_waiting_patients[iteration][current_time] = sim._fetal_waiting_patients()
        nonfetal_waiting_patients[iteration][current_time] = sim._nonfetal_waiting_patients()
        available_fetal_room[current_time] += (len([room for room in sim.state['rooms'] if
                                                       room['fetal'] in ['yes', 'yes (reserved)'] and room[
                                                           'status'] == 'available']))/365
        available_nonfetal_room[current_time] += (
            len([room for room in sim.state['rooms'] if room['fetal'] == 'no' and room['status'] == 'available']))/365
        available_fetal_sonographer[current_time] += (
            len([sonographer for sonographer in sim.state['sonographers'] if
                 sonographer['fetal'] in ['yes', 'yes (reserved)'] and sonographer['status'] == 'available']))/365
        available_nonfetal_sonographer[current_time] += (
            len([sonographer for sonographer in sim.state['sonographers'] if
                 sonographer['fetal'] == 'no' and sonographer['status'] == 'available']))/365
        current_time += 1

# Plot the resources
# Define the start time as 8:00 AM and end time as 7:00 PM
start_time = datetime.strptime('08:00', '%H:%M')
end_time = datetime.strptime('19:00', '%H:%M')

# Generate datetime objects for each time step from 8:00 to 19:00
time_steps_datetime = [start_time + timedelta(minutes=i) for i in range((end_time - start_time).seconds // 60 )]

# Add the end time (19:00) to the time steps
time_steps_datetime.append(end_time)

fs=20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)


# Plot number of available fetal rooms
plt.plot(time_steps_datetime, available_fetal_room[:len(time_steps_datetime)], label='Available Fetal Rooms', color='blue')

# Plot number of available fetal sonographer
plt.plot(time_steps_datetime, available_fetal_sonographer[:len(time_steps_datetime)], label='Available Fetal Sonographers', color='orange')

# Plot number of available nonfetal rooms
plt.plot(time_steps_datetime, available_nonfetal_room[:len(time_steps_datetime)], label='Available Nonfetal Rooms', color='green')

# Plot number of available nonfetal sonographers
plt.plot(time_steps_datetime, available_nonfetal_sonographer[:len(time_steps_datetime)], label='Available Nonfetal Sonographers', color='red')

# Set x-axis format to show time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Set x-axis limits to include the entire range of time steps
plt.xlim(start_time, end_time)

# Set labels and title
plt.xticks(rotation=45)
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Average Count', fontsize=fs)
plt.legend(fontsize=fs-7)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.title('Mean of Available Echo Resources', fontsize=fs)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', '1 resources.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()


# Plot the waiting patients
mean_ontime_waiting_patients = []
sd_ontime_waiting_patients = []
for col_index in range(len(time_steps_datetime)):
    # Extract the data points for the current column across all 365 days
    column_data = [ontime_waiting_patients[day_index][col_index] for day_index in range(365)]
    # Compute the mean of the column data
    column_mean = statistics.mean(column_data)
    column_sd = statistics.stdev(column_data)
    # Append the mean and sd to the result list
    mean_ontime_waiting_patients.append(column_mean)
    sd_ontime_waiting_patients.append(column_sd)

mean_late_waiting_patients = []
sd_late_waiting_patients = []
for col_index in range(len(time_steps_datetime)):
    # Extract the data points for the current column across all 365 days
    column_data = [late_waiting_patients[day_index][col_index] for day_index in range(365)]
    # Compute the mean of the column data
    column_mean = statistics.mean(column_data)
    column_sd = statistics.stdev(column_data)
    # Append the mean and sd to the result list
    mean_late_waiting_patients.append(column_mean)
    sd_late_waiting_patients.append(column_sd)

mean_early_waiting_patients = []
sd_early_waiting_patients = []
for col_index in range(len(time_steps_datetime)):
    # Extract the data points for the current column across all 365 days
    column_data = [early_waiting_patients[day_index][col_index] for day_index in range(365)]
    # Compute the mean of the column data
    column_mean = statistics.mean(column_data)
    column_sd = statistics.stdev(column_data)
    # Append the mean and sd to the result list
    mean_early_waiting_patients.append(column_mean)
    sd_early_waiting_patients.append(column_sd)

mean_fetal_waiting_patients = []
sd_fetal_waiting_patients = []
for col_index in range(len(time_steps_datetime)):
    # Extract the data points for the current column across all 365 days
    column_data = [fetal_waiting_patients[day_index][col_index] for day_index in range(365)]
    # Compute the mean of the column data
    column_mean = statistics.mean(column_data)
    column_sd = statistics.stdev(column_data)
    # Append the mean and sd to the result list
    mean_fetal_waiting_patients.append(column_mean)
    sd_fetal_waiting_patients.append(column_sd)

mean_nonfetal_waiting_patients = []
sd_nonfetal_waiting_patients = []
for col_index in range(len(time_steps_datetime)):
    # Extract the data points for the current column across all 365 days
    column_data = [nonfetal_waiting_patients[day_index][col_index] for day_index in range(365)]
    # Compute the mean of the column data
    column_mean = statistics.mean(column_data)
    column_sd = statistics.stdev(column_data)
    # Append the mean and sd to the result list
    mean_nonfetal_waiting_patients.append(column_mean)
    sd_nonfetal_waiting_patients.append(column_sd)

fs=20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)


# Plot number of ontime waiting patients

plt.plot(time_steps_datetime, mean_ontime_waiting_patients[:len(time_steps_datetime)], label='ontime waiting patients', color='blue')

# Plot number of late waiting patients
plt.plot(time_steps_datetime, mean_late_waiting_patients[:len(time_steps_datetime)], label='late waiting patients', color='orange')

# Plot number of early waiting patients
plt.plot(time_steps_datetime, mean_early_waiting_patients[:len(time_steps_datetime)], label='early waiting patients', color='green')

# Plot number of fetal waiting patients
plt.plot(time_steps_datetime, mean_fetal_waiting_patients[:len(time_steps_datetime)], label='fetal waiting patients', color='red', linestyle='dashdot')

# Plot number of nonfetal waiting patients
plt.plot(time_steps_datetime, mean_nonfetal_waiting_patients[:len(time_steps_datetime)], label='nonfetal waiting patients', color='purple', linestyle='dashdot')

# Set x-axis format to show time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Set x-axis limits to include the entire range of time steps
plt.xlim(start_time, end_time)

# Set labels and title
plt.xticks(rotation=45)
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Average Count', fontsize=fs)
plt.legend(fontsize=fs-7)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.title('Mean of Waiting Patients', fontsize=fs)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', '1 patients mean.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()


fs=20
plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')
plt.rc('text', usetex=True)


# Plot number of ontime waiting patients

plt.plot(time_steps_datetime, sd_ontime_waiting_patients[:len(time_steps_datetime)], label='ontime waiting patients', color='blue')

# Plot number of late waiting patients
plt.plot(time_steps_datetime, sd_late_waiting_patients[:len(time_steps_datetime)], label='late waiting patients', color='orange')

# Plot number of early waiting patients
plt.plot(time_steps_datetime, sd_early_waiting_patients[:len(time_steps_datetime)], label='early waiting patients', color='green')

# Plot number of fetal waiting patients
plt.plot(time_steps_datetime, sd_fetal_waiting_patients[:len(time_steps_datetime)], label='fetal waiting patients', color='red', linestyle='dashdot')

# Plot number of nonfetal waiting patients
plt.plot(time_steps_datetime, sd_nonfetal_waiting_patients[:len(time_steps_datetime)], label='nonfetal waiting patients', color='purple', linestyle='dashdot')

# Set x-axis format to show time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Set x-axis limits to include the entire range of time steps
plt.xlim(start_time, end_time)

# Set labels and title
plt.xticks(rotation=45)
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Standard deviation', fontsize=fs)
plt.legend(fontsize=fs-7)
plt.tick_params(axis='both', which='both', labelsize=fs-2)
plt.title('Sd of Waiting Patients', fontsize=fs)
plt.grid(True)
plt.tight_layout()
pdf_path = os.path.join('/Users/bozhisun/Desktop/echo project', '1 patients sd.pdf')
plt.savefig(pdf_path, format='pdf')
plt.close()