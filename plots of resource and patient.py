from simulation import ontime_waiting_patients
from simulation import late_waiting_patients
from simulation import available_fetal_room
from simulation import available_nonfetal_room
from simulation import available_fetal_sonographer
from simulation import available_nonfetal_sonographer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Plot the resources
# Define the start time as 8:00 AM and end time as 7:00 PM
start_time = datetime.strptime('08:00', '%H:%M')
end_time = datetime.strptime('19:00', '%H:%M')

# Generate datetime objects for each time step from 8:00 to 19:00
time_steps_datetime = [start_time + timedelta(minutes=i) for i in range((end_time - start_time).seconds // 60 )]

# Add the end time (19:00) to the time steps
time_steps_datetime.append(end_time)

# Plotting
plt.figure(figsize=(10, 6))

# Plot number of available fetal rooms
plt.plot(time_steps_datetime, available_fetal_room[:len(time_steps_datetime)], label='Available Fetal Rooms', color='blue')

# Plot number of available fetal sonographer
# raphers
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
plt.xlabel('Time', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Number of Available Rooms and Sonographers over Time', fontsize=18)

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

# Plot the waiting patients
# Define the start time as 8:00 AM and end time as 7:00 PM
start_time = datetime.strptime('08:00', '%H:%M')
end_time = datetime.strptime('19:00', '%H:%M')

# Generate datetime objects for each time step from 8:00 to 19:00
time_steps_datetime = [start_time + timedelta(minutes=i) for i in range((end_time - start_time).seconds // 60 )]

# Add the end time (19:00) to the time steps
time_steps_datetime.append(end_time)

# Plotting
plt.figure(figsize=(10, 6))

# Plot number of ontime waiting patients
plt.plot(time_steps_datetime, ontime_waiting_patients[:len(time_steps_datetime)], label='Ontime waiting patients', color='blue')

# Plot number of late waiting patients
plt.plot(time_steps_datetime, late_waiting_patients[:len(time_steps_datetime)], label='Late waiting patients', color='green')

# Set x-axis format to show time
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

# Set x-axis limits to include the entire range of time steps
plt.xlim(start_time, end_time)

# Set labels and title
plt.xlabel('Time', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Number of waiting patients', fontsize=18)

# Add legend
plt.legend()
#
# Show plot
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()
