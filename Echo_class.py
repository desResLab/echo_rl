# import the packages
import re
import simpy
import numpy as np
import random
import math
from scipy.stats import gamma
from scipy.optimize import root_scalar
from scipy.integrate import quad

# create the Echo class for the simulation
class Echo:
    def __init__(self, time_start, time_close, num_fetal_room, num_nonfetal_room, num_sonographer_both,
                 num_sonographer_nonfetal, time_sonographer_break, rate_sonographer_leave, rate_absence,
                 ontime_anger, late_anger, render_env):
        self.time_start = time_start
        self.time_close = time_close
        self.num_fetal_room = num_fetal_room
        self.num_nonfetal_room = num_nonfetal_room
        self.num_sonographer_both = num_sonographer_both
        self.num_sonographer_nonfetal = num_sonographer_nonfetal
        self.time_sonographer_break = time_sonographer_break
        self.rate_sonographer_leave = rate_sonographer_leave
        self.rate_absence = rate_absence
        self.ontime_anger = ontime_anger
        self.late_anger = late_anger
        self.render_env = render_env
        self.state = {
            'patients': [],
            'sonographers': [],
            'rooms': []
        }
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    def convert_to_time(self, steps):
        start_hours, start_minutes = map(int, self.time_start.split(':'))
        hours = int((start_hours * 60 + start_minutes + steps) // 60)
        minutes = int((start_hours * 60 + start_minutes + steps) % 60)
        return "{:02d}:{:02d}".format(hours, minutes)

    def convert_to_step(self, time):
        hours, minutes = map(int, time.split(':'))
        step = hours * 60 + minutes - 480
        return step

    # create a function to store all the appointments of the day (by steps)
    def _load_schedules(self):
        # calculate the number of time steps based on the time range
        time_range = self.convert_to_step(self.time_close) - self.convert_to_step(self.time_start)
        # create a variable to track the patients
        num = 0
        # Iterate over each time step
        for time in range(time_range):
            # Check if it's time to schedule a fetal patient (assuming 60 is the interval)
            if (time - 15) % 60 == 0:
                self.state['patients'].append({'patient number': num,
                                               'status': 'incoming',
                                               # incoming, waiting (late,early,ontime), testing, done
                                               'Schedule time': time,
                                               'Arrival time': 'NA',
                                               'fetal': 'yes',
                                               'subspecialty': 'na'})
                num += 1
                # Check if it's time to schedule a nonfetal patient (assuming 30 is the interval)
            if time % 30 == 0:
                # Check specific intervals where there are two appointments
                if (self.convert_to_step('10:30') <= time <= self.convert_to_step('12:30')) \
                        or (self.convert_to_step('14:00') <= time <= self.convert_to_step('15:30')):
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'subspecialty': 'na'})
                    num += 1
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'subspecialty': 'na'})
                    num += 1
                # Otherwise there is one appointment scheduled
                else:
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'subspecialty': 'na'})
                    num += 1
            # Handle specific appointments made at different times
            if time in [self.convert_to_step('10:45'), self.convert_to_step('11:45')]:
                self.state['patients'].append({'patient number': num,
                                               'status': 'incoming',
                                               'Schedule time': time,
                                               'Arrival time': 'NA',
                                               'fetal': 'no',
                                               'subspecialty': 'na'})
                num += 1
        #Sort the appointments by 'Schedule time'
        appointments = sorted(self.state['patients'], key=lambda x: x['Schedule time'])
        # Update the state with the sorted appointments
        self.state['patients'] = appointments

    # create two functions to simulate the arrival time of nonfetal patients and fetal patients respectively
    def _arrival_time(self, patient):
        random_number = random.random()
        # get the schedule time of the patients
        schedule_steps = patient['Schedule time']
        # 20% patient arrives earlier
        # 80% patient arrives later than the schedule
        if random_number < 0.8:
            arrival_steps = np.random.exponential(20, size=1) + schedule_steps
        # 10% patient arrives earlier than the schedule
        else:
            def exponential_pdf(x, theta):
                return np.exp(-x / theta) / theta

            def cumulative_distribution_function(x, a, b, theta):
                cdf = quad(exponential_pdf, a, x, args=(theta,))[0] / quad(exponential_pdf, a, b, args=(theta,))[0]
                return cdf

            u = np.random.rand()  # Generate a uniform random number
            # Use root_scalar to find the root of the function
            early_step = root_scalar(lambda x, u=u: cumulative_distribution_function(x, 0, 60, 5) - u,
                                     bracket=[0, 60]).root
            arrival_steps = -early_step + schedule_steps
        # if the patient comes before the opening time of the hospital, record the arrival time as the openning time
        if arrival_steps < 0:
            arrival_steps = 0
        # convert the arrival_steps to an integer
        arrival_steps = math.ceil(arrival_steps)
        if arrival_steps == 0:
            patient['Arrival time'] = 0
            if arrival_steps - schedule_steps < -10:
                patient['status'] = 'waiting(early)'
            else:
                patient['status'] = 'waiting(ontime)'
        return arrival_steps


    def _record_arrival_time(self, patient, arrival_steps):
        # record the arrival time only if the patients come before closing time and not absent
        random_number = random.random()
        schedule_steps = patient['Schedule time']
        if arrival_steps > 0 and arrival_steps <= self.convert_to_step(self.time_close) and random_number > self.rate_absence:
           yield self.env.timeout(arrival_steps-1)
           patient['Arrival time'] = arrival_steps
           if arrival_steps - schedule_steps > 10:
              patient['status'] = 'waiting(late)'
           elif arrival_steps - schedule_steps < -10:
              patient['status'] = 'waiting(early)'
           else:
              patient['status'] = 'waiting(ontime)'

    # create a function to load the patients
    def _load_patients(self):
        # Iterate over each time step
        for patient in self.state['patients']:
            arrival_steps = self._arrival_time(patient)
            self.env.process(self._record_arrival_time(patient, arrival_steps))

    # load the sonographers
    def _load_sonographers(self):
        # single out a fetal sonographer as the reserved one for the future policy function and there is at least one
        # fetal sonographer and at least one non-fetal sonographer for the program to run
        self.state['sonographers'].append({'sonographer number': 0,
                                           'fetal': 'yes (reserved)',
                                           'status': 'available',  # available, break, leave, working, reserved
                                           'break times': 2})  # the remaining break times
        for i in range(self.num_sonographer_both - 1):
            if random.random() < 1-self.rate_sonographer_leave:  # 90% probability the sonographer can work that day
                status = 'available'
            else:
                status = 'leave'
            self.state['sonographers'].append({'sonographer number': i + 1,
                                               'fetal': 'yes',
                                               'status': status,  # available, break, leave, working, reserved
                                               'break times': 2})  # the remaining break times
        self.state['sonographers'].append({'sonographer number': self.num_sonographer_both,
                                           'fetal': 'no',
                                           'status': 'available',  # available, break, leave, working, reserved
                                           'break times': 2})  # the remaining break times
        for i in range(self.num_sonographer_nonfetal-1):
            if random.random() < 1-self.rate_sonographer_leave:  # 90% probability
                status = 'available'
            else:
                status = 'leave'
            self.state['sonographers'].append({'sonographer number': i + 1 + self.num_sonographer_both,
                                               'fetal': 'no',
                                               'status': status,
                                               'break times': 2})

    # load the echo rooms
    def _load_rooms(self):
        # single out an fetal echo room as the reserved one for the future policy function
        self.state['rooms'].append({'room number': 0,
                                    'fetal': 'yes (reserved)',
                                    'status': 'available'})  # available, break, leave, working, reserved
        for i in range(self.num_fetal_room - 1):
            self.state['rooms'].append({'room number': i + 1,
                                        'fetal': 'yes',
                                        'status': 'available'})  # available, working
        for i in range(self.num_nonfetal_room):
            self.state['rooms'].append({'room number': i + self.num_fetal_room,
                                        'fetal': 'no',
                                        'status': 'available'})

    # simulate the spell time of the echo test
    def _spell_time(self, sonographer_num, room_num, patient_num):
        def gamma_pdf(x, k, theta):
            return gamma.pdf(x, k, scale=theta)

        def cumulative_distribution_function(x, a, b, k, theta):
            integral, _ = quad(gamma_pdf, a, x, args=(k, theta))
            normalization_factor, _ = quad(gamma_pdf, a, b, args=(k, theta))
            return integral / normalization_factor

        u = np.random.rand()  # Generate a uniform random number
        spell_time = root_scalar(lambda x, u=u: cumulative_distribution_function(x, 20, 150, 10, 4.5) - u,
                                 bracket=[20, 150]).root
        yield self.env.timeout(spell_time)
        # change the status of the patient to 'done' and room to 'available' after the test
        self.state['patients'][patient_num]['status'] = 'done'
        self.state['rooms'][room_num]['status'] = 'available'
        if self.state['sonographers'][sonographer_num]['break times'] == 2:
            random_number = random.random()
            # 20% the sonographer takes two breaks together
            if random_number < 0.2:
                self.state['sonographers'][sonographer_num]['status'] = 'break'
                self.state['sonographers'][sonographer_num]['break times'] = 0
                yield self.env.timeout(self.time_sonographer_break * 2)
                self.state['sonographers'][sonographer_num]['status'] = 'available'
            # 30% the sonographer takes one break
            elif random_number > 0.2 and random_number < 0.5:
                self.state['sonographers'][sonographer_num]['status'] = 'break'
                self.state['sonographers'][sonographer_num]['break times'] = 1
                yield self.env.timeout(self.time_sonographer_break)
                self.state['sonographers'][sonographer_num]['status'] = 'available'
            # 50% that the sonographer doesn't take break
            else:
                self.state['sonographers'][sonographer_num]['status'] = 'available'
        # if the sonographer has one time left
        elif self.state['sonographers'][sonographer_num]['break times'] == 1:
            random_number = random.random()
            # 30% the sonographer takes one break
            if random_number < 0.3:
                self.state['sonographers'][sonographer_num]['status'] = 'break'
                self.state['sonographers'][sonographer_num]['break times'] = 0
                yield self.env.timeout(self.time_sonographer_break)
                self.state['sonographers'][sonographer_num]['status'] = 'available'
            else:
                self.state['sonographers'][sonographer_num]['status'] = 'available'
        # if the sonographer has no break left
        else:
            self.state['sonographers'][sonographer_num]['status'] = 'available'

    # policy function
    def _adjust_patients(self, action):
        # For patients arrive early, if the time reaches their schedule time, change their status to waiting on time
        # and the arrival time changes to the schedule time
        for patient in self.state['patients']:
            if patient['status'] == 'waiting(early)' and self.env.now - patient['Schedule time'] >= -10:
                patient['status'] = 'waiting(ontime)'
                patient['Arrival time'] = self.env.now
        #Create three lists to store patients arrive late, arrive ontime and arrive early
        patients_late = []
        patients_ontime = []
        patients_early = []
        for patient in self.state['patients']:
            if patient['status'] == 'waiting(late)':
                patients_late.append(patient)
            elif patient['status'] == 'waiting(ontime)':
                patients_ontime.append(patient)
            elif patient['status'] == 'waiting(early)':
                patients_early.append(patient)

        if action == 1:
            # store all the available sonographers who can do fetal test into a list
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                           if (sonographer['fetal'] in ['yes', 'yes (reserved)']) and (
                                                       sonographer['status'] == 'available')]
            # store all the available rooms for fetal tests into a list
            room_fetal_available = [room for room in self.state['rooms']
                                    if (room['fetal'] in ['yes', 'yes (reserved)']) and (room['status'] == 'available')]
            # store all the available sonographers
            sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                     if sonographer['status'] == 'available']
            # store all the available rooms
            room_available = [room for room in self.state['rooms']
                              if room['status'] == 'available']

            # deal with ontime waiting patients first
            if patients_ontime != []:
                while True:
                    if patients_ontime == []:
                        break
                    # single out the nonfetal patients,
                    nonfetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    # max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']))
                    max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x['Schedule time'] else (self.env.now - x['Schedule time']))
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                          key=lambda x: (self.env.now - x['Arrival time']) if self.env.now >x['Schedule time'] else (
                                                                      self.env.now - x['Schedule time']))
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_ontime.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_ontime.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break

            # Then deal with late waiting patients
            if patients_late != []:
                while True:
                    if patients_late == []:
                        break
                    # single out the nonfetal patients",
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                   key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_late.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break
        elif action == 2:
            # store all the available sonographers who can do fetal test into a list
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                           if (sonographer['fetal'] in ['yes', 'yes (reserved)']) and (
                                                       sonographer['status'] == 'available')]
            # store all the available rooms for fetal tests into a list
            room_fetal_available = [room for room in self.state['rooms']
                                    if (room['fetal'] in ['yes', 'yes (reserved)']) and (room['status'] == 'available')]
            # store all the available sonographers
            sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                     if sonographer['status'] == 'available']
            # store all the available rooms
            room_available = [room for room in self.state['rooms']
                              if room['status'] == 'available']

            # deal with ontime waiting patients first
            if patients_ontime != []:
                while True:
                    if patients_ontime == []:
                        break
                    # single out the nonfetal patients,
                    nonfetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if x['Arrival time'] > x['Schedule time'] else (self.env.now - x['Schedule time']))
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients, key=lambda x: (self.env.now - x['Arrival time']) if x['Arrival time'] > x['Schedule time'] else (self.env.now - x['Schedule time']))
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_ontime.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_ontime.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break

            # Then deal with late waiting patients
            if patients_late != []:
                while True:
                    if patients_late == []:
                        break
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                   key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_late.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break
            # Then deal with patients arrived early
            if patients_early != []:
                while True:
                    if patients_early == []:
                        break
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_early if
                                         patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_early,
                                              key=lambda x: self.env.now - x['Schedule time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_early.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                   key=lambda x: self.env.now - x[
                                                                       'Schedule time'])
                                sonographer = random.choice(
                                    [sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_early.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(
                                    self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_early.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break
        elif action == 3:
            # First deal with the patients arrived on time
            # get the reserved fetal sonographer
            # store all the available sonographers who can do fetal test into a list except the reserved one
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers'] if
                                           sonographer['fetal'] == 'yes' and sonographer['status'] == 'available']
            # store all the available rooms for fetal tests into a list except the reserved one
            room_fetal_available = [room for room in self.state['rooms'] if
                                    room['fetal'] == 'yes' and room['status'] == 'available']
            # store all the available sonographers except the reserved one for fetal
            sonographer_available = [sonographer for sonographer in self.state['sonographers'] if
                                     sonographer['fetal'] != 'yes (reserved)' and sonographer['status'] == 'available']
            # store all the available rooms except the reserved one for fetal
            room_available = [room for room in self.state['rooms'] if
                               room['fetal'] != 'yes (reserved)' and room['status'] == 'available']
            # deal with ontime waiting patients
            if patients_ontime != []:
                while patients_ontime != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if x['Arrival time'] > x['Schedule time'] else (self.env.now - x['Schedule time']))
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients, key=lambda x: (self.env.now - x['Arrival time']) if x['Arrival time'] > x['Schedule time'] else (self.env.now - x['Schedule time']))
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_ontime.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_ontime:
                                if patient['fetal'] == 'no':
                                    patients_ontime.remove(patient)
                        else:
                            break
            # Then deal with patients arrived late
            if patients_late != []:
                while patients_late != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_late if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # find the nonfetal patient has been waiting the longest time
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                    key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_late:
                                if patient['fetal'] == 'no':
                                    patients_late.remove(patient)
                        else:
                            break
        elif action == 4:
            # First deal with the patients arrived on time
            # get the reserved fetal sonographer
            # store all the available sonographers who can do fetal test into a list except the reserved one
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers'] if
                                           sonographer['fetal'] == 'yes' and sonographer['status'] == 'available']
            # store all the available rooms for fetal tests into a list except the reserved one
            room_fetal_available = [room for room in self.state['rooms'] if
                                    room['fetal'] == 'yes' and room['status'] == 'available']
            # store all the available sonographers except the reserved one for fetal
            sonographer_available = [sonographer for sonographer in self.state['sonographers'] if
                                     sonographer['fetal'] != 'yes (reserved)' and sonographer['status'] == 'available']
            # store all the available rooms except the reserved one for fetal
            room_available = [room for room in self.state['rooms'] if
                              room['fetal'] != 'yes (reserved)' and room['status'] == 'available']
            # deal with ontime waiting patients
            if patients_ontime != []:
                while patients_ontime != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if x['Arrival time'] > x['Schedule time'] else (self.env.now - x['Schedule time']))
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                          key=lambda x: (self.env.now - x['Arrival time']) if x[
                                                                                                                  'Arrival time'] >
                                                                                                              x[
                                                                                                                  'Schedule time'] else (
                                                                      self.env.now - x['Schedule time']))
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_ontime.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_ontime:
                                if patient['fetal'] == 'no':
                                    patients_ontime.remove(patient)
                        else:
                            break
            # Then deal with patients arrived late
            if patients_late != []:
                while patients_late != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_late if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                    key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_late:
                                if patient['fetal'] == 'no':
                                    patients_late.remove(patient)
                        else:
                            break
            if patients_early != []:
                while patients_early != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_early if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_early if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_early, key=lambda x: self.env.now - x['Schedule time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                    key=lambda x: self.env.now - x['Schedule time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_early.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_early.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_early:
                                if patient['fetal'] == 'no':
                                    patients_early.remove(patient)
                        else:
                            break
        elif action == 5:
            for sonographer in self.state['sonographers']:
                if 'reserved' in sonographer['status']:
                    match = re.search(r'reserved for (\d+)', sonographer['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    sonographer_num = sonographer['sonographer number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and self.state['patients'][patient_num]['status'] == 'incoming':
                       self.state['sonographers'][sonographer_num]['status'] = 'available'
            for room in self.state['rooms']:
                if 'reserved' in room['status']:
                    match = re.search(r'reserved for (\d+)', room['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    room_num = room['room number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and self.state['patients'][patient_num]['status'] == 'incoming':
                       self.state['rooms'][room_num]['status'] = 'available'

            # store all the available sonographers who can do fetal test into a list
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                           if (sonographer['fetal'] in ['yes', 'yes (reserved)']) and (
                                                   sonographer['status'] == 'available')]
            # store all the available rooms for fetal tests into a list
            room_fetal_available = [room for room in self.state['rooms']
                                    if (room['fetal'] in ['yes', 'yes (reserved)']) and (room['status'] == 'available')]
            # store all the available sonographers
            sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                     if sonographer['status'] == 'available']
            # store all the available rooms
            room_available = [room for room in self.state['rooms']
                              if room['status'] == 'available']
            patients_ontime_no_reservation=[]
            if patients_ontime != []:
              for patient in patients_ontime:
                  max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x['Schedule time'] else (self.env.now - x['Schedule time']))
                  patient_num = max_waiting_patient['patient number']
                  # if the patient has been waiting the longest time is fetal
                  if max_waiting_patient['fetal'] == 'yes':
                      sonographer_reservation = False
                      room_reservation = False
                      reserved_sonographer_num = -1
                      reserved_room_num = -1
                      for sonographer in self.state['sonographers']:
                          if 'reserved for ' + str(patient_num) == sonographer['status']:
                              sonographer_reservation = True
                              reserved_sonographer_num = sonographer['sonographer number']
                      for room in self.state['rooms']:
                          if 'reserved for ' + str(patient_num) == room['status']:
                              room_reservation = True
                              reserved_room_num = room['room number']
                      if sonographer_reservation == True and room_reservation == True:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == True and room_reservation == False and room_fetal_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          room = random.choice([room for room in room_fetal_available])
                          room_num = room['room number']
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                          room_fetal_available.remove(room)
                          room_available.remove(room)
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == False and room_reservation == True and sonographer_fetal_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                          sonographer_num = sonographer['sonographer number']
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                          sonographer_fetal_available.remove(sonographer)
                          sonographer_available.remove(sonographer)
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == False and room_reservation == False and room_fetal_available != [] and sonographer_fetal_available != []:
                          # get an available sonographer and room
                          sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                          sonographer_num = sonographer['sonographer number']
                          room = random.choice([room for room in room_fetal_available])
                          room_num = room['room number']
                          max_waiting_patient['status'] = 'testing'
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                          room_fetal_available.remove(room)
                          sonographer_fetal_available.remove(sonographer)
                          room_available.remove(room)
                          sonographer_available.remove(sonographer)
                      else:
                          if sonographer_reservation == False and room_reservation == False and (room_fetal_available != [] or sonographer_fetal_available != []) == True:
                              patients_ontime_no_reservation.append(max_waiting_patient)
                          patients_ontime.remove(max_waiting_patient)
                  elif max_waiting_patient['fetal'] == 'no':
                      sonographer_reservation = False
                      room_reservation = False
                      reserved_sonographer_num = -1
                      reserved_room_num = -1
                      for sonographer in self.state['sonographers']:
                          if 'reserved for ' + str(patient_num) == sonographer['status']:
                              sonographer_reservation = True
                              reserved_sonographer_num = sonographer['sonographer number']
                      for room in self.state['rooms']:
                          if 'reserved for ' + str(patient_num) == room['status']:
                              room_reservation = True
                              reserved_room_num = room['room number']
                      if sonographer_reservation == True and room_reservation == True:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == True and room_reservation == False and room_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          room = random.choice([room for room in room_available])
                          room_num = room['room number']
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                          room_available.remove(room)
                          if room['fetal'] in ['yes', 'yes (reserved)']:
                              room_fetal_available.remove(room)
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == False and room_reservation == True and sonographer_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          sonographer = random.choice([sonographer for sonographer in sonographer_available])
                          sonographer_num = sonographer['sonographer number']
                          self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          sonographer_available.remove(sonographer)
                          if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                              sonographer_fetal_available.remove(sonographer)
                          patients_ontime.remove(max_waiting_patient)
                      elif room_available != [] and sonographer_available != []:
                          # get an available sonographer and room
                          sonographer = random.choice([sonographer for sonographer in sonographer_available])
                          sonographer_num = sonographer['sonographer number']
                          room = random.choice([room for room in room_available])
                          room_num = room['room number']
                          max_waiting_patient['status'] = 'testing'
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                          room_available.remove(room)
                          if room['fetal'] in ['yes', 'yes (reserved)']:
                              room_fetal_available.remove(room)
                          sonographer_available.remove(sonographer)
                          if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                              sonographer_fetal_available.remove(sonographer)
                      else:
                          if sonographer_reservation == False and room_reservation == False and (room_available != [] or sonographer_available != []) == True:
                              patients_ontime_no_reservation.append(max_waiting_patient)
                          patients_ontime.remove(max_waiting_patient)
            if patients_ontime_no_reservation != []:
                patients_before_current_time = []
                patients_after_current_time = []

                for patient in patients_ontime_no_reservation:
                    if patient['Schedule time'] > self.env.now:
                        patients_after_current_time.append(patient)
                    else:
                        patients_before_current_time.append(patient)

                # Sort the list
                patients_before_current_time.sort(key=lambda x: x['Arrival time'])
                patients_after_current_time.sort(key=lambda x: x['Schedule time'])

                # Combine the lists, placing patients with schedule time after current time first
                patients_ontime_no_reservation = patients_before_current_time + patients_after_current_time
                for patient in patients_ontime_no_reservation:
                    if patient['fetal'] == 'yes':
                        if room_fetal_available:
                            room = random.choice(room_fetal_available)
                            room['status'] = 'reserved for ' + str(patient['patient number'])
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                        if sonographer_fetal_available:
                            sonographer = random.choice(sonographer_fetal_available)
                            sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                    elif patient['fetal'] == 'no':
                        if room_available:
                            room = random.choice(room_available)
                            room['status'] = 'reserved for ' + str(patient['patient number'])
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                        if sonographer_available:
                            sonographer = random.choice(sonographer_available)
                            sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)



            # Check for incoming patients and make reservations
            # check if there is any reserved room without reserved room
            # store all the available sonographers who can do fetal test into a list
            patients_before_current_time = []
            patients_after_current_time = []
            filtered_patients = [patient for patient in self.state['patients'] if patient.get('status') == 'incoming']
            for patient in filtered_patients:
                if patient['Schedule time'] > self.env.now:
                    patients_after_current_time.append(patient)
                else:
                    patients_before_current_time.append(patient)

            # Sort each list by the schedule time
            patients_before_current_time.sort(key=lambda x: x['Arrival time'])
            patients_after_current_time.sort(key=lambda x: x['Schedule time'])

            # Combine the lists, placing patients with schedule time after current time first
            filtered_patients = patients_before_current_time + patients_after_current_time



            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] <= self.env.now:
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_available:
                                        room = random.choice(room_fetal_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                elif patient['fetal'] == 'no':
                                    if room_available:
                                        room = random.choice(room_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_available.remove(room)
                                        if room['fetal'] in ['yes', 'yes (reserved)']:
                                            room_fetal_available.remove(room)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_available:
                                        sonographer = random.choice(sonographer_fetal_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(sonographer_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if (0 <= self.env.now - patient['Schedule time'] < 10) and (patient['status'] not in ['testing', 'done']) and (patient['patient number'] not in reserved_patient_num):
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                if patient['fetal'] == 'yes':
                    if room_fetal_available:
                        room = random.choice(room_fetal_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_available:
                        sonographer = random.choice(sonographer_fetal_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_fetal_available.remove(sonographer)
                        sonographer_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice(room_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] in ['yes', 'yes (reserved)']:
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice(sonographer_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                            sonographer_fetal_available.remove(sonographer)

            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] > self.env.now:
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_available:
                                        room = random.choice(room_fetal_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                elif patient['fetal'] == 'no':
                                    if room_available:
                                        room = random.choice(room_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_available.remove(room)
                                        if room['fetal'] in ['yes', 'yes (reserved)']:
                                            room_fetal_available.remove(room)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_available:
                                        sonographer = random.choice(sonographer_fetal_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(sonographer_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if (-10 <= self.env.now - patient['Schedule time'] < 0) and (patient['status'] not in ['testing', 'done']) and (patient['patient number'] not in reserved_patient_num):
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                if patient['fetal'] == 'yes':
                    if room_fetal_available:
                        room = random.choice(room_fetal_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_available:
                        sonographer = random.choice(sonographer_fetal_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_fetal_available.remove(sonographer)
                        sonographer_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice(room_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] in ['yes', 'yes (reserved)']:
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice(sonographer_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                            sonographer_fetal_available.remove(sonographer)

            # Then deal with late waiting patients
            if patients_late != []:
                while True:
                    if patients_late == []:
                        break
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                   key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_late.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break
        elif action == 6:
            for sonographer in self.state['sonographers']:
                if 'reserved' in sonographer['status']:
                    match = re.search(r'reserved for (\d+)', sonographer['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    sonographer_num = sonographer['sonographer number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and self.state['patients'][patient_num]['status'] == 'incoming':
                       self.state['sonographers'][sonographer_num]['status'] = 'available'
            for room in self.state['rooms']:
                if 'reserved' in room['status']:
                    match = re.search(r'reserved for (\d+)', room['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    room_num = room['room number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and self.state['patients'][patient_num]['status'] == 'incoming':
                       self.state['rooms'][room_num]['status'] = 'available'

            # store all the available sonographers who can do fetal test into a list
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                           if (sonographer['fetal'] in ['yes', 'yes (reserved)']) and (
                                                   sonographer['status'] == 'available')]
            # store all the available rooms for fetal tests into a list
            room_fetal_available = [room for room in self.state['rooms']
                                    if (room['fetal'] in ['yes', 'yes (reserved)']) and (room['status'] == 'available')]
            # store all the available sonographers
            sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                     if sonographer['status'] == 'available']
            # store all the available rooms
            room_available = [room for room in self.state['rooms']
                              if room['status'] == 'available']
            patients_ontime_no_reservation = []
            if patients_ontime != []:
              for patient in patients_ontime:
                  max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if x['Arrival time'] > x['Schedule time'] else (self.env.now - x['Schedule time']))
                  patient_num = max_waiting_patient['patient number']
                  # if the patient has been waiting the longest time is fetal
                  if max_waiting_patient['fetal'] == 'yes':
                      sonographer_reservation = False
                      room_reservation = False
                      reserved_sonographer_num = -1
                      reserved_room_num = -1
                      for sonographer in self.state['sonographers']:
                          if 'reserved for ' + str(patient_num) == sonographer['status']:
                              sonographer_reservation = True
                              reserved_sonographer_num = sonographer['sonographer number']
                      for room in self.state['rooms']:
                          if 'reserved for ' + str(patient_num) == room['status']:
                              room_reservation = True
                              reserved_room_num = room['room number']
                      if sonographer_reservation == True and room_reservation == True:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == True and room_reservation == False and room_fetal_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          room = random.choice([room for room in room_fetal_available])
                          room_num = room['room number']
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                          room_fetal_available.remove(room)
                          room_available.remove(room)
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == False and room_reservation == True and sonographer_fetal_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                          sonographer_num = sonographer['sonographer number']
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                          sonographer_fetal_available.remove(sonographer)
                          sonographer_available.remove(sonographer)
                          patients_ontime.remove(max_waiting_patient)
                      elif room_fetal_available != [] and sonographer_fetal_available != []:
                          # get an available sonographer and room
                          sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                          sonographer_num = sonographer['sonographer number']
                          room = random.choice([room for room in room_fetal_available])
                          room_num = room['room number']
                          max_waiting_patient['status'] = 'testing'
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                          room_fetal_available.remove(room)
                          sonographer_fetal_available.remove(sonographer)
                          room_available.remove(room)
                          sonographer_available.remove(sonographer)
                      else:
                          if sonographer_reservation == False and room_reservation == False and (
                                  room_fetal_available != [] or sonographer_fetal_available != []) == True:
                              patients_ontime_no_reservation.append(max_waiting_patient)
                          patients_ontime.remove(max_waiting_patient)
                  elif max_waiting_patient['fetal'] == 'no':
                      sonographer_reservation = False
                      room_reservation = False
                      reserved_sonographer_num = -1
                      reserved_room_num = -1
                      for sonographer in self.state['sonographers']:
                          if 'reserved for ' + str(patient_num) == sonographer['status']:
                              sonographer_reservation = True
                              reserved_sonographer_num = sonographer['sonographer number']
                      for room in self.state['rooms']:
                          if 'reserved for ' + str(patient_num) == room['status']:
                              room_reservation = True
                              reserved_room_num = room['room number']
                      if sonographer_reservation == True and room_reservation == True:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == True and room_reservation == False and room_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                          room = random.choice([room for room in room_available])
                          room_num = room['room number']
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                          room_available.remove(room)
                          if room['fetal'] in ['yes', 'yes (reserved)']:
                              room_fetal_available.remove(room)
                          patients_ontime.remove(max_waiting_patient)
                      elif sonographer_reservation == False and room_reservation == True and sonographer_available != []:
                          self.state['patients'][patient_num]['status'] = 'testing'
                          self.state['rooms'][reserved_room_num]['status'] = 'working'
                          sonographer = random.choice([sonographer for sonographer in sonographer_available])
                          sonographer_num = sonographer['sonographer number']
                          self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          sonographer_available.remove(sonographer)
                          if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                              sonographer_fetal_available.remove(sonographer)
                          patients_ontime.remove(max_waiting_patient)
                      elif room_available != [] and sonographer_available != []:
                          # get an available sonographer and room
                          sonographer = random.choice([sonographer for sonographer in sonographer_available])
                          sonographer_num = sonographer['sonographer number']
                          room = random.choice([room for room in room_available])
                          room_num = room['room number']
                          max_waiting_patient['status'] = 'testing'
                          self.state['sonographers'][sonographer_num]['status'] = 'working'
                          self.state['rooms'][room_num]['status'] = 'working'
                          self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                          patients_ontime.remove(max_waiting_patient)
                          room_available.remove(room)
                          if room['fetal'] in ['yes', 'yes (reserved)']:
                              room_fetal_available.remove(room)
                          sonographer_available.remove(sonographer)
                          if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                              sonographer_fetal_available.remove(sonographer)
                      else:
                          if sonographer_reservation == False and room_reservation == False and (
                                  room_available != [] or sonographer_available != []) == True:
                              patients_ontime_no_reservation.append(max_waiting_patient)
                          patients_ontime.remove(max_waiting_patient)
            if patients_ontime_no_reservation != []:
                patients_before_current_time = []
                patients_after_current_time = []

                for patient in patients_ontime_no_reservation:
                    if patient['Schedule time'] > self.env.now:
                        patients_after_current_time.append(patient)
                    else:
                        patients_before_current_time.append(patient)

                # Sort the list
                patients_before_current_time.sort(key=lambda x: x['Arrival time'])
                patients_after_current_time.sort(key=lambda x: x['Schedule time'])

                # Combine the lists, placing patients with schedule time after current time first
                patients_ontime_no_reservation = patients_before_current_time + patients_after_current_time
                for patient in patients_ontime_no_reservation:
                    if patient['fetal'] == 'yes':
                        if room_fetal_available:
                            room = random.choice(room_fetal_available)
                            room['status'] = 'reserved for ' + str(patient['patient number'])
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                        if sonographer_fetal_available:
                            sonographer = random.choice(sonographer_fetal_available)
                            sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                    elif patient['fetal'] == 'no':
                        if room_available:
                            room = random.choice(room_available)
                            room['status'] = 'reserved for ' + str(patient['patient number'])
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                        if sonographer_available:
                            sonographer = random.choice(sonographer_available)
                            sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)

            # Check for incoming patients and make reservations
            # check if there is any reserved room without reserved room
            # store all the available sonographers who can do fetal test into a list
            patients_before_current_time = []
            patients_after_current_time = []
            filtered_patients = [patient for patient in self.state['patients'] if patient.get('status') == 'incoming']
            for patient in filtered_patients:
                if patient['Schedule time'] > self.env.now:
                    patients_after_current_time.append(patient)
                else:
                    patients_before_current_time.append(patient)

            # Sort each list by the schedule time
            patients_before_current_time.sort(key=lambda x: x['Arrival time'])
            patients_after_current_time.sort(key=lambda x: x['Schedule time'])

            # Combine the lists, placing patients with schedule time after current time first
            filtered_patients = patients_before_current_time + patients_after_current_time

            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] <= self.env.now:
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in
                                       self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_available:
                                        room = random.choice(room_fetal_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                elif patient['fetal'] == 'no':
                                    if room_available:
                                        room = random.choice(room_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_available.remove(room)
                                        if room['fetal'] in ['yes', 'yes (reserved)']:
                                            room_fetal_available.remove(room)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in
                                       self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_available:
                                        sonographer = random.choice(sonographer_fetal_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(sonographer_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if (0 <= self.env.now - patient['Schedule time'] < 10) and (
                        patient['status'] not in ['testing', 'done']) and (
                        patient['patient number'] not in reserved_patient_num):
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                if patient['fetal'] == 'yes':
                    if room_fetal_available:
                        room = random.choice(room_fetal_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_available:
                        sonographer = random.choice(sonographer_fetal_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_fetal_available.remove(sonographer)
                        sonographer_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice(room_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] in ['yes', 'yes (reserved)']:
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice(sonographer_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                            sonographer_fetal_available.remove(sonographer)

            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] > self.env.now:
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in
                                       self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_available:
                                        room = random.choice(room_fetal_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                elif patient['fetal'] == 'no':
                                    if room_available:
                                        room = random.choice(room_available)
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room_available.remove(room)
                                        if room['fetal'] in ['yes', 'yes (reserved)']:
                                            room_fetal_available.remove(room)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in
                                       self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_available:
                                        sonographer = random.choice(sonographer_fetal_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(sonographer_available)
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if (-10 <= self.env.now - patient['Schedule time'] < 0) and (
                        patient['status'] not in ['testing', 'done']) and (
                        patient['patient number'] not in reserved_patient_num):
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                if patient['fetal'] == 'yes':
                    if room_fetal_available:
                        room = random.choice(room_fetal_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_available:
                        sonographer = random.choice(sonographer_fetal_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_fetal_available.remove(sonographer)
                        sonographer_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice(room_available)
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] in ['yes', 'yes (reserved)']:
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice(sonographer_available)
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                            sonographer_fetal_available.remove(sonographer)

            # Then deal with late waiting patients
            if patients_late != []:
                while True:
                    if patients_late == []:
                        break
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                   key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_late.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break
            # Then deal with early waiting patients
            if patients_early != []:
                while True:
                    if patients_early == []:
                        break
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_early if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_early, key=lambda x: self.env.now - x['Schedule time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if there is available fetal room and available sonographer
                        if room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_early.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                   key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_patient['patient number']
                                max_waiting_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_early.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                    sonographer_fetal_available.remove(sonographer)
                                room_available.remove(room)
                                if room['fetal'] in ['yes', 'yes (reserved)']:
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            nonfetal_patients.remove(max_waiting_patient)
                            patients_early.remove(max_waiting_patient)
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                        else:
                            break
        elif action == 7:
            for sonographer in self.state['sonographers']:
                if 'reserved' in sonographer['status']:
                    match = re.search(r'reserved for (\d+)', sonographer['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    sonographer_num = sonographer['sonographer number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and \
                            self.state['patients'][patient_num]['status'] == 'incoming':
                        self.state['sonographers'][sonographer_num]['status'] = 'available'
            for room in self.state['rooms']:
                if 'reserved' in room['status']:
                    match = re.search(r'reserved for (\d+)', room['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    room_num = room['room number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and \
                            self.state['patients'][patient_num]['status'] == 'incoming':
                        self.state['rooms'][room_num]['status'] = 'available'
            # store all the available sonographers who can do fetal test into a list except the reserved one
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                            if sonographer['fetal'] == 'yes' and sonographer[
                                                'status'] == 'available']
            # store all the available rooms for fetal tests into a list except the reserved one
            room_fetal_available = [room for room in self.state['rooms']
                                    if room['fetal'] == 'yes' and room['status'] == 'available']
            # store all the available sonographers except the reserved one for fetal
            sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                    if sonographer['fetal'] != 'yes (reserved)' and sonographer['status'] == 'available']
            # store all the available rooms except the reserved one for fetal
            room_available = [room for room in self.state['rooms']
                                if room['fetal'] != 'yes (reserved)' and room['status'] == 'available']
            patients_ontime_no_reservation = []
            # First deal with the patients arrived on time
            if patients_ontime != []:
                for patient in patients_ontime:
                    # Find the patient who has been waiting the longest time
                    max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x['Schedule time'] else (self.env.now - x['Schedule time']))
                    patient_num = max_waiting_patient['patient number']
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                       sonographer['fetal'] == 'yes (reserved)'), None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                        (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                        None)
                    # If the patient who has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        sonographer_reservation = False
                        room_reservation = False
                        reserved_sonographer_num = -1
                        reserved_room_num = -1

                        for sonographer in self.state['sonographers']:
                            if 'reserved for ' + str(patient_num) == sonographer['status']:
                                sonographer_reservation = True
                                reserved_sonographer_num = sonographer['sonographer number']

                        for room in self.state['rooms']:
                            if 'reserved for ' + str(patient_num) == room['status']:
                                room_reservation = True
                                reserved_room_num = room['room number']

                        if sonographer_reservation == True and room_reservation == True:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == True and room_reservation == False and room_fetal_reserved[
                            'status'] == 'available':
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            room_num = room_fetal_reserved['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == True and room_reservation == False and room_fetal_available !=[]:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            room = random.choice(room_fetal_available)
                            room_num = room['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == False and room_reservation == True and sonographer_fetal_reserved[
                            'status'] == 'available':
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == False and room_reservation == True and sonographer_fetal_available!= []:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            sonographer = random.choice(sonographer_fetal_available)
                            sonographer_num = sonographer['sonographer number']
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_ontime.remove(max_waiting_patient)

                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_reserved['status'] == 'available':
                            self.state['patients'][patient_num]['status'] = 'testing'
                            room_num = room_fetal_reserved['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif (room_fetal_available != [] and sonographer_fetal_available != []):
                            sonographer = random.choice(sonographer_fetal_available)
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice(room_fetal_available)
                            room_num = room['room number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        else:
                            if sonographer_reservation == False and room_reservation == False and (
                                    room_fetal_available != [] or sonographer_fetal_available != []) == True:
                                patients_ontime_no_reservation.append(max_waiting_patient)
                            patients_ontime.remove(max_waiting_patient)

                    elif max_waiting_patient['fetal'] == 'no':
                        sonographer_reservation = False
                        room_reservation = False
                        reserved_sonographer_num = -1
                        reserved_room_num = -1
                        for sonographer in self.state['sonographers']:
                            if 'reserved for ' + str(patient_num) == sonographer['status']:
                                sonographer_reservation = True
                                reserved_sonographer_num = sonographer['sonographer number']
                        for room in self.state['rooms']:
                            if 'reserved for ' + str(patient_num) == room['status']:
                                room_reservation = True
                                reserved_room_num = room['room number']
                        if sonographer_reservation == True and room_reservation == True:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                        elif sonographer_reservation == True and room_reservation == False and room_available != []:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                        elif sonographer_reservation == False and room_reservation == True and sonographer_available != []:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            patients_ontime.remove(max_waiting_patient)
                        elif room_available != [] and sonographer_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                        else:
                            if sonographer_reservation == False and room_reservation == False and (
                                    room_available != [] or sonographer_available != []) == True:
                                patients_ontime_no_reservation.append(max_waiting_patient)
                            patients_ontime.remove(max_waiting_patient)

                if patients_ontime_no_reservation != []:
                    patients_before_current_time = []
                    patients_after_current_time = []

                    for patient in patients_ontime_no_reservation:
                        if patient['Schedule time'] > self.env.now:
                            patients_after_current_time.append(patient)
                        else:
                            patients_before_current_time.append(patient)

                    # Sort the list
                    patients_before_current_time.sort(key=lambda x: x['Arrival time'])
                    patients_after_current_time.sort(key=lambda x: x['Schedule time'])

                    # Combine the lists, placing patients with schedule time after current time first
                    patients_ontime_no_reservation = patients_before_current_time + patients_after_current_time
                    for patient in patients_ontime_no_reservation:
                        if patient['fetal'] == 'yes':
                            if room_fetal_reserved['status'] == 'available':
                                room_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                            elif room_fetal_available:
                                room = random.choice([room for room in room_fetal_available])
                                room['status'] = 'reserved for ' + str(patient['patient number'])
                                room['fetal'] = 'yes (reserved)'
                                room_fetal_reserved['fetal'] = 'yes'
                                room_fetal_available.remove(room)
                                room_available.remove(room)
                            if sonographer_fetal_reserved['status'] == 'available':
                                sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                            elif sonographer_fetal_available:
                                sonographer = random.choice(
                                    [sonographer for sonographer in sonographer_fetal_available])
                                sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                                sonographer['fetal'] = 'yes (reserved)'
                                sonographer_fetal_reserved['fetal'] = 'yes'
                                sonographer_available.remove(sonographer)
                                sonographer_fetal_available.remove(sonographer)
                        elif patient['fetal'] == 'no':
                            if room_available:
                                room = random.choice([room for room in room_available])
                                room['status'] = 'reserved for ' + str(patient['patient number'])
                                room_available.remove(room)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                            if sonographer_available:
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)

            patients_before_current_time = []
            patients_after_current_time = []

            for patient in self.state['patients']:
                if patient['Schedule time'] > self.env.now:
                    patients_after_current_time.append(patient)
                else:
                    patients_before_current_time.append(patient)

            # Sort each list by the schedule time
            patients_before_current_time.sort(key=lambda x: x['Schedule time'])
            patients_after_current_time.sort(key=lambda x: x['Schedule time'])

            # Combine the lists, placing patients with schedule time after current time first
            self.state['patients'] = patients_before_current_time + patients_after_current_time
            # Check for incoming patients and make reservations
            # check if there is any reserved room without reserved room
            # get the reserved fetal sonographer
            filtered_patients = [patient for patient in self.state['patients'] if patient.get('status') == 'incoming']
            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] <= self.env.now:
                    patient_num = patient['patient number']
                    sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                       sonographer['fetal'] == 'yes (reserved)'), None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                        (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                        None)
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_reserved['status'] == 'available':
                                        room_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif room_fetal_available:
                                        room = random.choice([room for room in room_fetal_available])
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room['fetal'] = 'yes (reserved)'
                                        room_fetal_reserved['fetal'] = 'yes'
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                        room_fetal_reserved = room
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_reserved['status'] == 'available':
                                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif sonographer_fetal_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_fetal_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer['fetal'] = 'yes (reserved)'
                                        sonographer_fetal_reserved['fetal'] = 'yes'
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                        sonographer_fetal_reserved = sonographer
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if 0 <= self.env.now - patient['Schedule time'] < 10 and patient['status'] not in ['testing', 'done'] and \
                        patient['patient number'] not in reserved_patient_num:
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                   sonographer['fetal'] == 'yes (reserved)'), None)
                # get the reserved fetal room
                room_fetal_reserved = next(
                    (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                    None)
                if patient['fetal'] == 'yes':
                    if room_fetal_reserved['status'] == 'available':
                        room_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif room_fetal_available:
                        room = random.choice([room for room in room_fetal_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room['fetal'] = 'yes (reserved)'
                        room_fetal_reserved['fetal'] = 'yes'
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_reserved['status'] == 'available':
                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif sonographer_fetal_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer['fetal'] = 'yes (reserved)'
                        sonographer_fetal_reserved['fetal'] = 'yes'
                        sonographer_available.remove(sonographer)
                        sonographer_fetal_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice([room for room in room_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] == 'yes':
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] == 'yes':
                            sonographer_fetal_available.remove(sonographer)
            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] > self.env.now:
                    patient_num = patient['patient number']
                    sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                       sonographer['fetal'] == 'yes (reserved)'), None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                        (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                        None)
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_reserved['status'] == 'available':
                                        room_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif room_fetal_available:
                                        room = random.choice([room for room in room_fetal_available])
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room['fetal'] = 'yes (reserved)'
                                        room_fetal_reserved['fetal'] = 'yes'
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                        room_fetal_reserved = room
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_reserved['status'] == 'available':
                                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif sonographer_fetal_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_fetal_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer['fetal'] = 'yes (reserved)'
                                        sonographer_fetal_reserved['fetal'] = 'yes'
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                        sonographer_fetal_reserved = sonographer
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if -10 <= self.env.now - patient['Schedule time'] < 0 and patient['status'] not in ['testing', 'done'] and \
                        patient['patient number'] not in reserved_patient_num:
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                   sonographer['fetal'] == 'yes (reserved)'), None)
                # get the reserved fetal room
                room_fetal_reserved = next(
                    (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                    None)
                if patient['fetal'] == 'yes':
                    if room_fetal_reserved['status'] == 'available':
                        room_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif room_fetal_available:
                        room = random.choice([room for room in room_fetal_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room['fetal'] = 'yes (reserved)'
                        room_fetal_reserved['fetal'] = 'yes'
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_reserved['status'] == 'available':
                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif sonographer_fetal_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer['fetal'] = 'yes (reserved)'
                        sonographer_fetal_reserved['fetal'] = 'yes'
                        sonographer_available.remove(sonographer)
                        sonographer_fetal_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice([room for room in room_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] == 'yes':
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] == 'yes':
                            sonographer_fetal_available.remove(sonographer)

            # Then deal with patients arrived late
            if patients_late != []:
                while patients_late != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_late if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # find the nonfetal patient has been waiting the longest time
                    if nonfetal_patients != []:
                        max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                           key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                    key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_late:
                                if patient['fetal'] == 'no':
                                    patients_late.remove(patient)
                        else:
                            break
        elif action == 8:
            for sonographer in self.state['sonographers']:
                if 'reserved' in sonographer['status']:
                    match = re.search(r'reserved for (\d+)', sonographer['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    sonographer_num = sonographer['sonographer number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and \
                            self.state['patients'][patient_num]['status'] == 'incoming':
                        self.state['sonographers'][sonographer_num]['status'] = 'available'
            for room in self.state['rooms']:
                if 'reserved' in room['status']:
                    match = re.search(r'reserved for (\d+)', room['status'])
                    # Extract the number from the match
                    patient_num = int(match.group(1))
                    room_num = room['room number']
                    if self.env.now - self.state['patients'][patient_num]['Schedule time'] >= 10 and \
                            self.state['patients'][patient_num]['status'] == 'incoming':
                        self.state['rooms'][room_num]['status'] = 'available'
            # store all the available sonographers who can do fetal test into a list except the reserved one
            sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                            if sonographer['fetal'] == 'yes' and sonographer[
                                                'status'] == 'available']
            # store all the available rooms for fetal tests into a list except the reserved one
            room_fetal_available = [room for room in self.state['rooms']
                                    if room['fetal'] == 'yes' and room['status'] == 'available']
            # store all the available sonographers except the reserved one for fetal
            sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                    if sonographer['fetal'] != 'yes (reserved)' and sonographer['status'] == 'available']
            # store all the available rooms except the reserved one for fetal
            room_available = [room for room in self.state['rooms']
                                if room['fetal'] != 'yes (reserved)' and room['status'] == 'available']
            patients_ontime_no_reservation=[]
            # First deal with the patients arrived on time
            if patients_ontime != []:
                for patient in patients_ontime:
                    # Find the patient who has been waiting the longest time
                    max_waiting_patient = max(patients_ontime, key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x['Schedule time'] else (self.env.now - x['Schedule time']))
                    patient_num = max_waiting_patient['patient number']
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                       sonographer['fetal'] == 'yes (reserved)'), None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                        (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                        None)
                    # If the patient who has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        sonographer_reservation = False
                        room_reservation = False
                        reserved_sonographer_num = -1
                        reserved_room_num = -1

                        for sonographer in self.state['sonographers']:
                            if 'reserved for ' + str(patient_num) == sonographer['status']:
                                sonographer_reservation = True
                                reserved_sonographer_num = sonographer['sonographer number']

                        for room in self.state['rooms']:
                            if 'reserved for ' + str(patient_num) == room['status']:
                                room_reservation = True
                                reserved_room_num = room['room number']

                        if sonographer_reservation == True and room_reservation == True:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == True and room_reservation == False and room_fetal_reserved[
                            'status'] == 'available':
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            room_num = room_fetal_reserved['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == True and room_reservation == False and room_fetal_available !=[]:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            room = random.choice(room_fetal_available)
                            room_num = room['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == False and room_reservation == True and sonographer_fetal_reserved[
                            'status'] == 'available':
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif sonographer_reservation == False and room_reservation == True and sonographer_fetal_available!= []:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            sonographer = random.choice(sonographer_fetal_available)
                            sonographer_num = sonographer['sonographer number']
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_ontime.remove(max_waiting_patient)

                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_reserved['status'] == 'available':
                            self.state['patients'][patient_num]['status'] = 'testing'
                            room_num = room_fetal_reserved['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)

                        elif (room_fetal_available != [] and sonographer_fetal_available != []):
                            sonographer = random.choice(sonographer_fetal_available)
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice(room_fetal_available)
                            room_num = room['room number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            room_fetal_available.remove(room)
                            sonographer_fetal_available.remove(sonographer)
                            room_available.remove(room)
                            sonographer_available.remove(sonographer)
                        else:
                            if sonographer_reservation == False and room_reservation == False and (
                                    room_fetal_available != [] or sonographer_fetal_available != []) == True:
                                patients_ontime_no_reservation.append(max_waiting_patient)
                            patients_ontime.remove(max_waiting_patient)

                    elif max_waiting_patient['fetal'] == 'no':
                        sonographer_reservation = False
                        room_reservation = False
                        reserved_sonographer_num = -1
                        reserved_room_num = -1
                        for sonographer in self.state['sonographers']:
                            if 'reserved for ' + str(patient_num) == sonographer['status']:
                                sonographer_reservation = True
                                reserved_sonographer_num = sonographer['sonographer number']
                        for room in self.state['rooms']:
                            if 'reserved for ' + str(patient_num) == room['status']:
                                room_reservation = True
                                reserved_room_num = room['room number']
                        if sonographer_reservation == True and room_reservation == True:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, reserved_room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                        elif sonographer_reservation == True and room_reservation == False and room_available != []:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['sonographers'][reserved_sonographer_num]['status'] = 'working'
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(reserved_sonographer_num, room_num, patient_num))
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            patients_ontime.remove(max_waiting_patient)
                        elif sonographer_reservation == False and room_reservation == True and sonographer_available != []:
                            self.state['patients'][patient_num]['status'] = 'testing'
                            self.state['rooms'][reserved_room_num]['status'] = 'working'
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            self.env.process(self._spell_time(sonographer_num, reserved_room_num, patient_num))
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                            patients_ontime.remove(max_waiting_patient)
                        elif room_available != [] and sonographer_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_available])
                            room_num = room['room number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_ontime.remove(max_waiting_patient)
                            room_available.remove(room)
                            if room['fetal'] in ['yes', 'yes (reserved)']:
                                room_fetal_available.remove(room)
                            sonographer_available.remove(sonographer)
                            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                                sonographer_fetal_available.remove(sonographer)
                        else:
                            if sonographer_reservation == False and room_reservation == False and (
                                    room_available != [] or sonographer_available != []) == True:
                                patients_ontime_no_reservation.append(max_waiting_patient)
                            patients_ontime.remove(max_waiting_patient)

                if patients_ontime_no_reservation != []:
                    patients_before_current_time = []
                    patients_after_current_time = []

                    for patient in patients_ontime_no_reservation:
                        if patient['Schedule time'] > self.env.now:
                            patients_after_current_time.append(patient)
                        else:
                            patients_before_current_time.append(patient)

                    # Sort the list
                    patients_before_current_time.sort(key=lambda x: x['Arrival time'])
                    patients_after_current_time.sort(key=lambda x: x['Schedule time'])

                    # Combine the lists, placing patients with schedule time after current time first
                    patients_ontime_no_reservation = patients_before_current_time + patients_after_current_time
                    for patient in patients_ontime_no_reservation:
                        if patient['fetal'] == 'yes':
                            if room_fetal_reserved['status'] == 'available':
                                room_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                            elif room_fetal_available:
                                room = random.choice([room for room in room_fetal_available])
                                room['status'] = 'reserved for ' + str(patient['patient number'])
                                room['fetal'] = 'yes (reserved)'
                                room_fetal_reserved['fetal'] = 'yes'
                                room_fetal_available.remove(room)
                                room_available.remove(room)
                            if sonographer_fetal_reserved['status'] == 'available':
                                sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                            elif sonographer_fetal_available:
                                sonographer = random.choice(
                                    [sonographer for sonographer in sonographer_fetal_available])
                                sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                                sonographer['fetal'] = 'yes (reserved)'
                                sonographer_fetal_reserved['fetal'] = 'yes'
                                sonographer_available.remove(sonographer)
                                sonographer_fetal_available.remove(sonographer)
                        elif patient['fetal'] == 'no':
                            if room_available:
                                room = random.choice([room for room in room_available])
                                room['status'] = 'reserved for ' + str(patient['patient number'])
                                room_available.remove(room)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                            if sonographer_available:
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                                sonographer_available.remove(sonographer)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)

            patients_before_current_time = []
            patients_after_current_time = []

            for patient in self.state['patients']:
                if patient['Schedule time'] > self.env.now:
                    patients_after_current_time.append(patient)
                else:
                    patients_before_current_time.append(patient)

            # Sort each list by the schedule time
            patients_before_current_time.sort(key=lambda x: x['Schedule time'])
            patients_after_current_time.sort(key=lambda x: x['Schedule time'])

            # Combine the lists, placing patients with schedule time after current time first
            self.state['patients'] = patients_before_current_time + patients_after_current_time
            # Check for incoming patients and make reservations
            # check if there is any reserved room without reserved room
            # get the reserved fetal sonographer
            filtered_patients = [patient for patient in self.state['patients'] if patient.get('status') == 'incoming']
            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] <= self.env.now:
                    patient_num = patient['patient number']
                    sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                       sonographer['fetal'] == 'yes (reserved)'), None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                        (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                        None)
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_reserved['status'] == 'available':
                                        room_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif room_fetal_available:
                                        room = random.choice([room for room in room_fetal_available])
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room['fetal'] = 'yes (reserved)'
                                        room_fetal_reserved['fetal'] = 'yes'
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                        room_fetal_reserved = room
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_reserved['status'] == 'available':
                                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif sonographer_fetal_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_fetal_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer['fetal'] = 'yes (reserved)'
                                        sonographer_fetal_reserved['fetal'] = 'yes'
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                        sonographer_fetal_reserved = sonographer
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if 0 <= self.env.now - patient['Schedule time'] < 10 and patient['status'] not in ['testing', 'done'] and \
                        patient['patient number'] not in reserved_patient_num:
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                   sonographer['fetal'] == 'yes (reserved)'), None)
                # get the reserved fetal room
                room_fetal_reserved = next(
                    (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                    None)
                if patient['fetal'] == 'yes':
                    if room_fetal_reserved['status'] == 'available':
                        room_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif room_fetal_available:
                        room = random.choice([room for room in room_fetal_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room['fetal'] = 'yes (reserved)'
                        room_fetal_reserved['fetal'] = 'yes'
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_reserved['status'] == 'available':
                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif sonographer_fetal_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer['fetal'] = 'yes (reserved)'
                        sonographer_fetal_reserved['fetal'] = 'yes'
                        sonographer_available.remove(sonographer)
                        sonographer_fetal_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice([room for room in room_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] == 'yes':
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] == 'yes':
                            sonographer_fetal_available.remove(sonographer)
            reserved_patient_num = []
            for patient in filtered_patients:
                if patient['Schedule time'] > self.env.now:
                    patient_num = patient['patient number']
                    sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                       sonographer['fetal'] == 'yes (reserved)'), None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                        (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                        None)
                    patient_num = patient['patient number']
                    for sonographer in self.state['sonographers']:
                        if 'reserved for ' + str(patient_num) == sonographer['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == room['status'] for room in self.state['rooms']):
                                if patient['fetal'] == 'yes':
                                    if room_fetal_reserved['status'] == 'available':
                                        room_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif room_fetal_available:
                                        room = random.choice([room for room in room_fetal_available])
                                        room['status'] = 'reserved for ' + str(patient_num)
                                        room['fetal'] = 'yes (reserved)'
                                        room_fetal_reserved['fetal'] = 'yes'
                                        room_fetal_available.remove(room)
                                        room_available.remove(room)
                                        room_fetal_reserved = room
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
                    for room in self.state['rooms']:
                        if 'reserved for ' + str(patient_num) == room['status']:
                            reserved_patient_num.append(patient_num)
                            if not any(('reserved for ' + str(patient_num)) == sonographer['status'] for sonographer in self.state['sonographers']):
                                if patient['fetal'] == 'yes':
                                    if sonographer_fetal_reserved['status'] == 'available':
                                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient_num)
                                    elif sonographer_fetal_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_fetal_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer['fetal'] = 'yes (reserved)'
                                        sonographer_fetal_reserved['fetal'] = 'yes'
                                        sonographer_fetal_available.remove(sonographer)
                                        sonographer_available.remove(sonographer)
                                        sonographer_fetal_reserved = sonographer
                                elif patient['fetal'] == 'no':
                                    if sonographer_available:
                                        sonographer = random.choice(
                                            [sonographer for sonographer in sonographer_available])
                                        sonographer['status'] = 'reserved for ' + str(patient_num)
                                        sonographer_available.remove(sonographer)
                                        if sonographer['fetal'] == 'yes':
                                            sonographer_fetal_available.remove(sonographer)
            new_incoming_patients = []
            for patient in filtered_patients:
                if -10 <= self.env.now - patient['Schedule time'] < 0 and patient['status'] not in ['testing', 'done'] and \
                        patient['patient number'] not in reserved_patient_num:
                    new_incoming_patients.append(patient)
            for patient in new_incoming_patients:
                sonographer_fetal_reserved = next((sonographer for sonographer in self.state['sonographers'] if
                                                   sonographer['fetal'] == 'yes (reserved)'), None)
                # get the reserved fetal room
                room_fetal_reserved = next(
                    (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                    None)
                if patient['fetal'] == 'yes':
                    if room_fetal_reserved['status'] == 'available':
                        room_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif room_fetal_available:
                        room = random.choice([room for room in room_fetal_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room['fetal'] = 'yes (reserved)'
                        room_fetal_reserved['fetal'] = 'yes'
                        room_fetal_available.remove(room)
                        room_available.remove(room)
                    if sonographer_fetal_reserved['status'] == 'available':
                        sonographer_fetal_reserved['status'] = 'reserved for ' + str(patient['patient number'])
                    elif sonographer_fetal_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_fetal_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer['fetal'] = 'yes (reserved)'
                        sonographer_fetal_reserved['fetal'] = 'yes'
                        sonographer_available.remove(sonographer)
                        sonographer_fetal_available.remove(sonographer)
                elif patient['fetal'] == 'no':
                    if room_available:
                        room = random.choice([room for room in room_available])
                        room['status'] = 'reserved for ' + str(patient['patient number'])
                        room_available.remove(room)
                        if room['fetal'] == 'yes':
                            room_fetal_available.remove(room)
                    if sonographer_available:
                        sonographer = random.choice([sonographer for sonographer in sonographer_available])
                        sonographer['status'] = 'reserved for ' + str(patient['patient number'])
                        sonographer_available.remove(sonographer)
                        if sonographer['fetal'] == 'yes':
                            sonographer_fetal_available.remove(sonographer)
            # Then deal with patients arrived late
            if patients_late != []:
                while patients_late != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_late if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_late, key=lambda x: self.env.now - x['Arrival time'])
                    # find the nonfetal patient has been waiting the longest time
                    if nonfetal_patients != []:
                        max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                           key=lambda x: self.env.now - x['Arrival time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                    key=lambda x: self.env.now - x['Arrival time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_late.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_late.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_late:
                                if patient['fetal'] == 'no':
                                    patients_late.remove(patient)
                        else:
                            break
            if patients_early != []:
                while patients_early != []:
                    # get the reserved fetal sonographer
                    sonographer_fetal_reserved = next(
                            (sonographer for sonographer in self.state['sonographers'] if
                             sonographer['fetal'] == 'yes (reserved)'),
                            None)
                    # get the reserved fetal room
                    room_fetal_reserved = next(
                            (room for room in self.state['rooms'] if room['fetal'] == 'yes (reserved)'),
                            None)
                    # single out the fetal patients
                    fetal_patients = [patient for patient in patients_early if patient['fetal'] == 'yes']
                    # single out the nonfetal patients
                    nonfetal_patients = [patient for patient in patients_early if patient['fetal'] == 'no']
                    # find the patient has been waiting the longest time
                    max_waiting_patient = max(patients_early, key=lambda x: self.env.now - x['Schedule time'])
                    # if the patient has been waiting the longest time is fetal
                    if max_waiting_patient['fetal'] == 'yes':
                        # if the reserved sonographer and room are available
                        if sonographer_fetal_reserved['status'] == 'available' and room_fetal_reserved[
                            'status'] == 'available':
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room_num = room_fetal_reserved['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal sonographer is available, but the the reserved fetal room isn't available
                        elif sonographer_fetal_reserved['status'] == 'available' and room_fetal_available != []:
                            sonographer_num = sonographer_fetal_reserved['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if the reserved fetal room is available, but the the reserved fetal sonographer isn't available
                        elif room_fetal_reserved['status'] == 'available' and sonographer_fetal_available != []:
                            room_num = room_fetal_reserved['room number']
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        # if there is available fetal room and available sonographer but none of the reserved is available
                        elif room_fetal_available != [] and sonographer_fetal_available != []:
                            # get an available sonographer and room
                            sonographer = random.choice(
                                [sonographer for sonographer in sonographer_fetal_available])
                            sonographer_num = sonographer['sonographer number']
                            room = random.choice([room for room in room_fetal_available])
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            sonographer_fetal_reserved['fetal'] = 'yes'
                            room_fetal_reserved['fetal'] = 'yes'
                            self.state['sonographers'][sonographer_num]['fetal'] = 'yes (reserved)'
                            self.state['rooms'][room_num]['fetal'] = 'yes (reserved)'
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_fetal_available.remove(sonographer)
                            sonographer_available.remove(sonographer)
                            room_fetal_available.remove(room)
                            room_available.remove(room)
                            patients_early.remove(max_waiting_patient)
                            fetal_patients.remove(max_waiting_patient)
                        elif nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                            while nonfetal_patients != [] and room_available != [] and sonographer_available != []:
                                # find the nonfetal patient has been waiting the longest time
                                max_waiting_nonfetal_patient = max(nonfetal_patients,
                                                                    key=lambda x: self.env.now - x['Schedule time'])
                                sonographer = random.choice([sonographer for sonographer in sonographer_available])
                                sonographer_num = sonographer['sonographer number']
                                room = random.choice([room for room in room_available])
                                room_num = room['room number']
                                patient_num = max_waiting_nonfetal_patient['patient number']
                                max_waiting_nonfetal_patient['status'] = 'testing'
                                self.state['sonographers'][sonographer_num]['status'] = 'working'
                                self.state['rooms'][room_num]['status'] = 'working'
                                nonfetal_patients.remove(max_waiting_nonfetal_patient)
                                patients_early.remove(max_waiting_nonfetal_patient)
                                sonographer_available.remove(sonographer)
                                room_available.remove(room)
                                if sonographer['fetal'] == 'yes':
                                    sonographer_fetal_available.remove(sonographer)
                                if room['fetal'] == 'yes':
                                    room_fetal_available.remove(room)
                                self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            break
                        else:
                            break
                    elif max_waiting_patient['fetal'] == 'no':
                        if room_available != [] and sonographer_available != []:
                            sonographer = random.choice([sonographer for sonographer in sonographer_available])
                            room = random.choice([room for room in room_available])
                            sonographer_num = sonographer['sonographer number']
                            room_num = room['room number']
                            patient_num = max_waiting_patient['patient number']
                            max_waiting_patient['status'] = 'testing'
                            self.state['sonographers'][sonographer_num]['status'] = 'working'
                            self.state['rooms'][room_num]['status'] = 'working'
                            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
                            sonographer_available.remove(sonographer)
                            room_available.remove(room)
                            if sonographer['fetal']=='yes':
                                sonographer_fetal_available.remove(sonographer)
                            if room['fetal']=='yes':
                                room_fetal_available.remove(room)
                            patients_early.remove(max_waiting_patient)
                            nonfetal_patients.remove(max_waiting_patient)
                        elif fetal_patients != []:
                            for patient in patients_early:
                                if patient['fetal'] == 'no':
                                    patients_early.remove(patient)
                        else:
                            break
    def _ontime_waiting_patients(self):
        ontime_patients = [patient for patient in self.state['patients']
                            if patient['status'] in ['waiting(ontime)']]
        num = len(ontime_patients)
        return num

    # the function to record the number of waiting patients
    def _late_waiting_patients(self):
        late_patients = [patient for patient in self.state['patients']
                          if patient['status'] == 'waiting(late)']
        num = len(late_patients)
        return num

    # the function to render the states
    def render(self):
        ontime_waiting_patients = self._ontime_waiting_patients()
        late_waiting_patients = self._late_waiting_patients()
        print(f"time: {self.env.now}, \n\n"
              f"patients: {self.state['patients']}, \n\n"
              f"sonographers: {self.state['sonographers']}, \n\n"
              f"rooms: {self.state['rooms']}, \n\n"
              f"ontime waiting patients: {ontime_waiting_patients}, \n\n"
              f"late waiting patients: {late_waiting_patients}, \n\n")

        # the function to reset
    def reset(self):
        self.env = simpy.Environment()
        self.state = {
            'patients': [],
            'sonographers': [],
            'rooms': []
        }
        self._load_schedules()
        self._load_patients()
        self._load_sonographers()
        self._load_rooms()
        observations = [v for k, v in self.state.items()]
        return observations

    # penalty calculation
    def _penalty_calculation(self):
        penalty = 0
        patients_waiting_ontime = []
        patients_waiting_late = []
        for patient in self.state['patients']:
            if patient['status'] == 'waiting(ontime)':
                patients_waiting_ontime.append(patient)
            elif patient['status'] == 'waiting(late)':
                patients_waiting_late.append(patient)
        if self.env.now > self.convert_to_step(self.time_close):
            penalty += 5 * (len(patients_waiting_ontime) + len(patients_waiting_late))
        for patient in patients_waiting_ontime:
            if patient['Arrival time'] >= patient['Schedule time']:
                penalty += self.ontime_anger * (self.env.now - patient['Arrival time'])
            elif self.env.now > patient['Schedule time']:
                penalty += self.ontime_anger * (self.env.now - patient['Schedule time'])
        for patient in patients_waiting_late:
            if patient['fetal'] == 'yes':
                penalty += self.late_anger * (self.env.now - patient['Arrival time'])
        return penalty

        # the function to simulate the steps


    def step(self, action, render):
        self._adjust_patients(action)
        observations = [v for k, v in self.state.items()]
        ontime_waiting_patients = self._ontime_waiting_patients()
        late_waiting_patients = self._late_waiting_patients()
        terminal = True if self.env.now > self.convert_to_step(self.time_close) + 120 else False
        if render:
            self.render()
        return observations, ontime_waiting_patients, late_waiting_patients, terminal

