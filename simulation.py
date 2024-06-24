from Echo_class import Echo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
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
ontime_anger = 2
late_anger = 0
render_env = True
policy_index = 7
sim = Echo(time_start=time_start, time_close=time_close, num_fetal_room=num_fetal_room,
            num_nonfetal_room=num_nonfetal_room, num_sonographer_both=num_sonographer_both,
            num_sonographer_nonfetal=num_sonographer_nonfetal, time_sonographer_break=time_sonographer_break,
            rate_sonographer_leave=rate_sonographer_leave, rate_absence=rate_absence,
            ontime_anger=ontime_anger, late_anger=late_anger, render_env=render_env)
rows = sim.convert_to_step(time_close) + 121
ontime_waiting_patients = [0] * rows
late_waiting_patients = [0] * rows
available_fetal_room = [0] * rows
available_nonfetal_room = [0] * rows
available_fetal_sonographer = [0] * rows
available_nonfetal_sonographer = [0] * rows
current_time = sim.convert_to_step(time_start)
penalty = [[0] * rows for _ in range(9)]

for iteration in range(365):
    current_time = 0  # Reset current time for each policy
    sim.reset()  # Reset simulation for each policy
    while current_time <= sim.convert_to_step(time_close) + 120 :
        sim.step(policy_index, render_env )  # Apply the current policy
        ontime_waiting_patients[current_time] += sim._ontime_waiting_patients()/365
        late_waiting_patients[current_time] += sim._late_waiting_patients()/365
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
        sim.env.run(until=current_time + 1)
        current_time += 1


