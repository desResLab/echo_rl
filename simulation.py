from Echo_class import Echo
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import numpy as np
import random
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
render_env = True
policy_index = 5
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
penalty = [[0] * rows for _ in range(15)]



