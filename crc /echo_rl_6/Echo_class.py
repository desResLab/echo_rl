# import the packages
import simpy
import numpy as np
import random
import math
from scipy.stats import gamma
from scipy.optimize import root_scalar
from scipy.integrate import quad

num_action = 3
# create the Echo class for the simulation
class Echo:
    def __init__(self, time_start, time_close, num_fetal_room, num_nonfetal_room, num_sonographer_both,
                 num_sonographer_nonfetal, time_sonographer_break, rate_sonographer_leave, rate_absence,
                 render_env):
        self.time_start = time_start
        self.time_close = time_close
        self.num_fetal_room = num_fetal_room
        self.num_nonfetal_room = num_nonfetal_room
        self.num_sonographer_both = num_sonographer_both
        self.num_sonographer_nonfetal = num_sonographer_nonfetal
        self.time_sonographer_break = time_sonographer_break
        self.rate_sonographer_leave = rate_sonographer_leave
        self.rate_absence = rate_absence
        self.render_env = render_env
        self.reserved_fetal_pair = []
        self.reserved_nonfetal_pair = []
        self.state = {
            'patients': [],
            'sonographers': [],
            'rooms': []
        }
        self.unreserved_room = []
        self.unreserved_sonographer = []

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
                                               'reservation': [],
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
                                                   'reservation': [],
                                                   'subspecialty': 'na'})
                    num += 1
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'reservation': [],
                                                   'subspecialty': 'na'})
                    num += 1
                # Otherwise there is one appointment scheduled
                else:
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'reservation': [],
                                                   'subspecialty': 'na'})
                    num += 1
            # Handle specific appointments made at different times
            if time in [self.convert_to_step('10:45'), self.convert_to_step('11:45')]:
                self.state['patients'].append({'patient number': num,
                                               'status': 'incoming',
                                               'Schedule time': time,
                                               'Arrival time': 'NA',
                                               'fetal': 'no',
                                               'reservation': [],
                                               'subspecialty': 'na'})
                num += 1
        #Sort the appointments by 'Schedule time'
        appointments = sorted(self.state['patients'], key=lambda x: x['Schedule time'])
        # Update the state with the sorted appointments
        self.state['patients'] = appointments

    # create two functions to simulate the arrival time of patients
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

            U = np.random.rand()  # Generate a uniform random number
            # Use root_scalar to find the root of the function
            early_step = root_scalar(lambda x, u=U: cumulative_distribution_function(x, 0, 60, 5) - u,
                                     bracket=[0, 60]).root
            arrival_steps = -early_step + schedule_steps
        # if the patient comes before the opening time of the hospital, record the arrival time as the opening time
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

        U = np.random.rand()  # Generate a uniform random number
        spell_time = root_scalar(lambda x, u=U: cumulative_distribution_function(x, 20, 150, 10, 4.5) - u,
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

    # define a function to make reservations
    def _make_reservations(self, fetal_reservation_ratio, nonfetal_reservation_ratio):
        # define a function to make reservations
        # calculate the number of echo resources in the system
        num_fetal_room = 0
        num_nonfetal_room = 0
        num_both_sonographer = 0
        num_nonfetal_sonographer = 0
        for room in self.state['rooms']:
            if room['fetal'] in ['yes', 'yes (reserved)']:
                num_fetal_room += 1
        for room in self.state['rooms']:
            if room['fetal'] == 'no':
                num_nonfetal_room += 1
        for sonographer in self.state['sonographers']:
            if sonographer['fetal'] in ['yes', 'yes (reserved)'] and sonographer['status'] != 'leave':
                num_both_sonographer += 1
        for sonographer in self.state['sonographers']:
            if sonographer['fetal'] == 'no' and sonographer['status'] != 'leave':
                num_nonfetal_sonographer += 1
        # number of pairs the fetal and non-fetal resources can make
        num_fetal_pair = 0
        num_nonfetal_pair = 0
        if num_fetal_room < num_both_sonographer:
            num_fetal_pair = num_fetal_room
            num_nonfetal_pair = min(num_both_sonographer - num_fetal_room + num_nonfetal_sonographer,
                                    num_nonfetal_room)
        else:
            num_fetal_pair = num_both_sonographer
            num_nonfetal_pair = min(num_fetal_room - num_both_sonographer + num_nonfetal_room,
                                    num_nonfetal_sonographer)
        # change the number of reservations
        num_reserved_fetal_pair = num_fetal_pair * fetal_reservation_ratio
        num_reserved_nonfetal_pair = math.ceil(num_nonfetal_pair * nonfetal_reservation_ratio)
        reserved_room = []
        reserved_sonographer = []
        for i in range(num_fetal_pair):
            for room in self.state['rooms']:
                if (room['fetal'] in ['yes', 'yes (reserved)']) and (room not in reserved_room) and (
                        len(reserved_room) < num_fetal_pair):
                    reserved_room.append(room)
                    self.unreserved_room.remove(room)
                    for sonographer in self.state['sonographers']:
                        if (sonographer['fetal'] in ['yes', 'yes (reserved)']) and (
                                sonographer not in reserved_sonographer) and (sonographer['status'] != 'leave'):
                            reserved_sonographer.append(sonographer)
                            self.unreserved_sonographer.remove(sonographer)
                            self.reserved_fetal_pair.append(
                                [room['room number'], sonographer['sonographer number']])
                            break
        for i in range(num_reserved_nonfetal_pair):
            for room in self.state['rooms']:
                if (room not in reserved_room) and (
                        len(reserved_room) < num_fetal_pair + num_reserved_nonfetal_pair):
                    reserved_room.append(room)
                    self.unreserved_room.remove(room)
                    for sonographer in self.state['sonographers']:
                        if (sonographer not in reserved_sonographer) and (sonographer['status'] != 'leave'):
                            reserved_sonographer.append(sonographer)
                            self.unreserved_sonographer.remove(sonographer)
                            self.reserved_nonfetal_pair.append(
                                [room['room number'], sonographer['sonographer number']])
                            break
        while len(self.reserved_fetal_pair) > num_reserved_fetal_pair:
            self.unreserved_room.append(
                self.state['rooms'][self.reserved_fetal_pair[len(self.reserved_fetal_pair) - 1][0]])
            self.unreserved_sonographer.append(
                self.state['sonographers'][self.reserved_fetal_pair[len(self.reserved_fetal_pair) - 1][1]])
            self.reserved_fetal_pair.remove(self.reserved_fetal_pair[len(self.reserved_fetal_pair) - 1])
        fetal_patients = []
        nonfetal_patients = []
        for patient in self.state['patients']:
            if patient['fetal'] == 'yes':
                fetal_patients.append(patient)
            if patient['fetal'] == 'no':
                nonfetal_patients.append(patient)
        reserved_fetal_patients = fetal_patients.copy()[0:num_reserved_fetal_pair]
        reserved_nonfetal_patients = nonfetal_patients.copy()[0:num_reserved_nonfetal_pair]
        compared_index = 0
        if len(reserved_fetal_patients) != 0:
            for i in range(num_reserved_fetal_pair, len(fetal_patients)):
                if fetal_patients[i]['Schedule time'] - reserved_fetal_patients[compared_index][
                    'Schedule time'] >= 40:
                    reserved_fetal_patients.append(fetal_patients[i])
                    compared_index += 1

        compared_index = 0
        if len(reserved_nonfetal_patients) != 0:
            for i in range(num_reserved_nonfetal_pair, len(nonfetal_patients)):
                if nonfetal_patients[i]['Schedule time'] - reserved_nonfetal_patients[compared_index][
                    'Schedule time'] >= 40:
                    reserved_nonfetal_patients.append(nonfetal_patients[i])
                    compared_index += 1

        i = 0
        if num_reserved_fetal_pair != 0:
            for patient in reserved_fetal_patients:
                index = i % num_reserved_fetal_pair
                patient['reservation'] = self.reserved_fetal_pair[index]
                i += 1
        i = 0
        if num_reserved_nonfetal_pair != 0:
            for patient in reserved_nonfetal_patients:
                index = i % num_reserved_nonfetal_pair
                patient['reservation'] = self.reserved_nonfetal_pair[index]
                i += 1

    def _action(self,action_int):
        for patient in self.state['patients']:
            if patient['status'] == 'waiting(early)' and self.env.now - patient['Schedule time'] >= -10:
                patient['status'] = 'waiting(ontime)'
                patient['Arrival time'] = self.env.now
        action = [0] * 6
        action[0] = action_int//(num_action**5)
        action_int = action_int % (num_action**5)
        action[1] = action_int//(num_action**4)
        action_int = action_int % (num_action**4)
        action[2] = action_int//(num_action**3)
        action_int = action_int % (num_action**3)
        action[3] = action_int//(num_action**2)
        action_int = action_int % (num_action**2)
        action[4] = action_int//(num_action**1)
        action_int = action_int % (num_action**1)
        action[5] = action_int
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
        sonographer_fetal_available = [sonographer for sonographer in self.state['sonographers']
                                       if (sonographer['fetal'] in ['yes', 'yes (reserved)']) and (
                                               sonographer['status'] == 'available')]
        room_fetal_available = [room for room in self.state['rooms']
                                if (room['fetal'] in ['yes', 'yes (reserved)']) and (room['status'] == 'available')]
        sonographer_available = [sonographer for sonographer in self.state['sonographers']
                                 if sonographer['status'] == 'available']
        room_available = [room for room in self.state['rooms']
                          if room['status'] == 'available']
        for i in range(action[0]):
            fetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'yes']
            max_waiting_patient = max(fetal_patients,
                                      key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x[
                                          'Schedule time'] else (self.env.now - x['Schedule time']))
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
            fetal_patients.remove(max_waiting_patient)
            room_fetal_available.remove(room)
            sonographer_fetal_available.remove(sonographer)
            room_available.remove(room)
            sonographer_available.remove(sonographer)
        for i in range(action[1]):
            fetal_patients = [patient for patient in patients_late if patient['fetal'] == 'yes']
            max_waiting_patient = max(fetal_patients,
                                      key=lambda x: self.env.now - x['Arrival time'])
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
            fetal_patients.remove(max_waiting_patient)
            room_fetal_available.remove(room)
            sonographer_fetal_available.remove(sonographer)
            room_available.remove(room)
            sonographer_available.remove(sonographer)
        for i in range(action[2]):
            fetal_patients = [patient for patient in patients_early if patient['fetal'] == 'yes']
            max_waiting_patient = max(fetal_patients,
                                      key=lambda x: self.env.now - x['Schedule time'])
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
            fetal_patients.remove(max_waiting_patient)
            room_fetal_available.remove(room)
            sonographer_fetal_available.remove(sonographer)
            room_available.remove(room)
            sonographer_available.remove(sonographer)
        for i in range(action[3]):
            nonfetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'no']
            max_waiting_patient = max(nonfetal_patients,
                                      key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x[
                                          'Schedule time'] else (self.env.now - x['Schedule time']))
            nonfetal_room_available = []
            nonfetal_sonographer_available = []
            for room in room_available:
                if room['fetal'] == 'no':
                    nonfetal_room_available.append(room)
            for sonographer in sonographer_available:
                if sonographer['fetal'] == 'no':
                    nonfetal_sonographer_available.append(sonographer)
            if nonfetal_room_available != []:
                room = random.choice([room for room in nonfetal_room_available])
            else:
                room = random.choice([room for room in room_available])
            if nonfetal_sonographer_available != []:
                sonographer = random.choice([sonographer for sonographer in nonfetal_sonographer_available])
            else:
                sonographer = random.choice([sonographer for sonographer in sonographer_available])
            sonographer_num = sonographer['sonographer number']
            room_num = room['room number']
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            self.state['sonographers'][sonographer_num]['status'] = 'working'
            self.state['rooms'][room_num]['status'] = 'working'
            nonfetal_patients.remove(max_waiting_patient)
            patients_ontime.remove(max_waiting_patient)
            sonographer_available.remove(sonographer)
            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                sonographer_fetal_available.remove(sonographer)
            room_available.remove(room)
            if room['fetal'] in ['yes', 'yes (reserved)']:
                room_fetal_available.remove(room)
            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
        for i in range(action[4]):
            nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
            max_waiting_patient = max(nonfetal_patients,
                                      key=lambda x: self.env.now - x['Arrival time'])
            nonfetal_room_available = []
            nonfetal_sonographer_available = []
            for room in room_available:
                if room['fetal'] == 'no':
                    nonfetal_room_available.append(room)
            for sonographer in sonographer_available:
                if sonographer['fetal'] == 'no':
                    nonfetal_sonographer_available.append(sonographer)
            if nonfetal_room_available != []:
                room = random.choice([room for room in nonfetal_room_available])
            else:
                room = random.choice([room for room in room_available])
            if nonfetal_sonographer_available != []:
                sonographer = random.choice([sonographer for sonographer in nonfetal_sonographer_available])
            else:
                sonographer = random.choice([sonographer for sonographer in sonographer_available])
            sonographer_num = sonographer['sonographer number']
            room_num = room['room number']
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            self.state['sonographers'][sonographer_num]['status'] = 'working'
            self.state['rooms'][room_num]['status'] = 'working'
            nonfetal_patients.remove(max_waiting_patient)
            patients_late.remove(max_waiting_patient)
            sonographer_available.remove(sonographer)
            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                sonographer_fetal_available.remove(sonographer)
            room_available.remove(room)
            if room['fetal'] in ['yes', 'yes (reserved)']:
                room_fetal_available.remove(room)
            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))
        for i in range(action[5]):
            nonfetal_patients = [patient for patient in patients_early if patient['fetal'] == 'no']
            max_waiting_patient = max(nonfetal_patients,
                                      key=lambda x: self.env.now - x['Schedule time'])
            nonfetal_room_available = []
            nonfetal_sonographer_available = []
            for room in room_available:
                if room['fetal'] == 'no':
                    nonfetal_room_available.append(room)
            for sonographer in sonographer_available:
                if sonographer['fetal'] == 'no':
                    nonfetal_sonographer_available.append(sonographer)
            if nonfetal_room_available != []:
                room = random.choice([room for room in nonfetal_room_available])
            else:
                room = random.choice([room for room in room_available])
            if nonfetal_sonographer_available != []:
                sonographer = random.choice([sonographer for sonographer in nonfetal_sonographer_available])
            else:
                sonographer = random.choice([sonographer for sonographer in sonographer_available])
            sonographer_num = sonographer['sonographer number']
            room_num = room['room number']
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            self.state['sonographers'][sonographer_num]['status'] = 'working'
            self.state['rooms'][room_num]['status'] = 'working'
            nonfetal_patients.remove(max_waiting_patient)
            patients_early.remove(max_waiting_patient)
            sonographer_available.remove(sonographer)
            if sonographer['fetal'] in ['yes', 'yes (reserved)']:
                sonographer_fetal_available.remove(sonographer)
            room_available.remove(room)
            if room['fetal'] in ['yes', 'yes (reserved)']:
                room_fetal_available.remove(room)
            self.env.process(self._spell_time(sonographer_num, room_num, patient_num))

    # the function to record the number of waiting patients
    def _ontime_waiting_fetal_patients(self):
        ontime_waiting_fetal_patients = [patient for patient in self.state['patients']
                            if patient['status'] in ['waiting(ontime)'] and patient['fetal'] in ['yes']]
        num = len(ontime_waiting_fetal_patients)
        return num

    def _ontime_waiting_nonfetal_patients(self):
        ontime_waiting_nonfetal_patients = [patient for patient in self.state['patients']
                            if patient['status'] in ['waiting(ontime)'] and patient['fetal'] in ['no']]
        num = len(ontime_waiting_nonfetal_patients)
        return num

    def _ontime_waiting_patients(self):
        return self._ontime_waiting_fetal_patients()+self._ontime_waiting_nonfetal_patients()

    def _late_waiting_fetal_patients(self):
        late_waiting_fetal_patients = [patient for patient in self.state['patients']
                                         if patient['status'] in ['waiting(late)'] and patient['fetal'] in ['yes']]
        num = len(late_waiting_fetal_patients)
        return num

    def _late_waiting_nonfetal_patients(self):
        late_waiting_nonfetal_patients = [patient for patient in self.state['patients']
                                            if patient['status'] in ['waiting(late)'] and patient['fetal'] in ['no']]
        num = len(late_waiting_nonfetal_patients)
        return num

    def _late_waiting_patients(self):
        return self._late_waiting_fetal_patients() + self._late_waiting_nonfetal_patients()

    def _early_waiting_fetal_patients(self):
        early_waiting_fetal_patients = [patient for patient in self.state['patients']
                                         if patient['status'] in ['waiting(early)'] and patient['fetal'] in ['yes']]
        num = len(early_waiting_fetal_patients)
        return num

    def _early_waiting_nonfetal_patients(self):
        early_waiting_nonfetal_patients = [patient for patient in self.state['patients']
                                            if patient['status'] in ['waiting(early)'] and patient['fetal'] in ['no']]
        num = len(early_waiting_nonfetal_patients)
        return num

    def _early_waiting_patients(self):
        return self._early_waiting_fetal_patients() + self._early_waiting_nonfetal_patients()

    def _fetal_waiting_patients(self):
        return self._ontime_waiting_fetal_patients()+self._late_waiting_fetal_patients()+self._early_waiting_fetal_patients()

    def _nonfetal_waiting_patients(self):
        return self._ontime_waiting_nonfetal_patients()+self._late_waiting_nonfetal_patients()+self._early_waiting_nonfetal_patients()

    def _available_fetal_sonograpphers(self):
        available_fetal_sonograpphers = [sonographer for sonographer in self.state['sonographers']
                                            if sonographer['status'] in ['available'] and sonographer['fetal'] in ['yes', 'yes (reserved)']]
        num = len(available_fetal_sonograpphers)
        return num

    def _available_nonfetal_sonograpphers(self):
        available_nonfetal_sonograpphers = [sonographer for sonographer in self.state['sonographers']
                                            if sonographer['status'] in ['available'] and sonographer['fetal'] in ['no']]
        num = len(available_nonfetal_sonograpphers)
        return num

    def _available_fetal_rooms(self):
        available_fetal_rooms = [room for room in self.state['rooms']
                                if room['status'] in ['available'] and room['fetal'] in ['yes', 'yes (reserved)']]
        num = len(available_fetal_rooms)
        return num

    def _available_nonfetal_rooms(self):
        available_nonfetal_rooms = [room for room in self.state['rooms']
                                     if room['status'] in ['available'] and room['fetal'] in ['no']]
        num = len(available_nonfetal_rooms)
        return num

    def _onbreak_fetal_sonograpphers(self):
        onbreak_fetal_sonograpphers = [sonographer for sonographer in self.state['sonographers']
                                            if sonographer['status'] in ['leave'] and sonographer['fetal'] in ['yes', 'yes (reserved)']]
        num = len(onbreak_fetal_sonograpphers)
        return num

    def _onbreak_nonfetal_sonograpphers(self):
        onbreak_nonfetal_sonograpphers = [sonographer for sonographer in self.state['sonographers']
                                            if sonographer['status'] in ['leave'] and sonographer['fetal'] in ['no']]
        num = len(onbreak_nonfetal_sonograpphers)
        return num

    def _convert_early_patients(self):
        for patient in self.state['patients']:
            if patient['status'] == 'waiting(early)' and self.env.now - patient['Schedule time'] >= -10:
                patient['status'] = 'waiting(ontime)'
                patient['Arrival time'] = self.env.now

    # the function to render the states
    def render(self):
        ontime_waiting_patients = self._ontime_waiting_patients()
        late_waiting_patients = self._late_waiting_patients()
        early_waiting_patients = self._early_waiting_patients()
        fetal_waiting_patients = self._fetal_waiting_patients()
        nonfetal_waiting_patients = self._nonfetal_waiting_patients()
        print(f"time: {self.env.now}, \n\n"
              f"patients: {self.state['patients']}, \n\n"
              f"sonographers: {self.state['sonographers']}, \n\n"
              f"rooms: {self.state['rooms']}, \n\n"
              f"ontime waiting patients: {ontime_waiting_patients}, \n\n"
              f"late waiting patients: {late_waiting_patients}, \n\n"
              f"early waiting patients: {early_waiting_patients}, \n\n"
              f"fetal waiting patients: {fetal_waiting_patients}, \n\n"
              f"nonfetal waiting patients: {nonfetal_waiting_patients}, \n\n"
              )

        # the function to reset
    def reset(self, seed=None, options=None, rate_sonographer_leave = None):
        """Reset the environment."""

        # Handle the seed for reproducibility
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            # Also set the seed for other random libraries if necessary
            random.seed(seed)

        # Reset the environment state
        self.env = simpy.Environment()
        # Reapply all parameters from initialization
        self.rate_sonographer_leave = rate_sonographer_leave

        # Update dynamic parameters
        self.env.rate_sonographer_leave = rate_sonographer_leave
        self.state = {
            'patients': [],
            'sonographers': [],
            'rooms': []
        }
        self.reserved_fetal_pair = []
        self.reserved_nonfetal_pair = []
        self._load_schedules()
        self._load_patients()
        self._load_sonographers()
        self._load_rooms()
        self.unreserved_room = self.state['rooms'].copy()
        self.unreserved_sonographer = self.state['sonographers'].copy()

        # Handle options if provided (e.g., allowing dynamic configuration)
        if options is not None:
            # Use options to customize the reset logic if necessary
            # For example, you could modify `time_start` or other parameters dynamically
            pass
        self._convert_early_patients()
        # Generate initial observation
        state = [
            self._ontime_waiting_fetal_patients(),
            self._late_waiting_fetal_patients(),
            self._early_waiting_fetal_patients(),
            self._ontime_waiting_nonfetal_patients(),
            self._late_waiting_nonfetal_patients(),
            self._early_waiting_nonfetal_patients(),
            self._available_fetal_sonograpphers(),
            self._available_nonfetal_sonograpphers(),
            self._available_fetal_rooms(),
            self._available_nonfetal_rooms(),
            self._onbreak_fetal_sonograpphers(),
            self._onbreak_nonfetal_sonograpphers(),
            self.env.now
        ]

        # Return the initial observation and an empty info dictionary
        return np.array(state, dtype=np.int32), {}


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
        if self.env.now >= self.convert_to_step(self.time_close):
            for patient in self.state['patients']:
                if 'waiting' in patient['status']:
                    penalty += 10
        for patient in patients_waiting_ontime:
            if self.env.now >= patient['Schedule time']:
                penalty += 2 * (self.env.now - patient['Arrival time'])
        for patient in patients_waiting_late:
            penalty += 1 * (self.env.now - patient['Arrival time'])
        return penalty


    def step(self, action_int):
        """Take an action and return the next state, reward, terminated, truncated, and info."""
        # Apply the action provided by the agent
        self._action(action_int)

        # Calculate the penalty after taking the action and updating the environment
        penalty = self._penalty_calculation()
        reward = (-1)*penalty  # The more penalty, the lower the reward

        # Check termination condition (e.g., the environment ends after time_close + 120 minutes)
        terminated = self.env.now >= self.convert_to_step(self.time_close) + 120

        # Check for truncation condition if applicable (not defined here, so left as False)
        truncated = False

       # Advance the environment by one time step
        self.env.run(until=self.env.now + 1)
        self._convert_early_patients()
       # Generate the next state (observation)
        state_next = [
            self._ontime_waiting_fetal_patients(),
            self._late_waiting_fetal_patients(),
            self._early_waiting_fetal_patients(),
            self._ontime_waiting_nonfetal_patients(),
            self._late_waiting_nonfetal_patients(),
            self._early_waiting_nonfetal_patients(),
            self._available_fetal_sonograpphers(),
            self._available_nonfetal_sonograpphers(),
            self._available_fetal_rooms(),
            self._available_nonfetal_rooms(),
            self._onbreak_fetal_sonograpphers(),
            self._onbreak_nonfetal_sonograpphers(),
            self.env.now
        ]

        # Return the new state, reward, termination flag, truncated flag, and any additional info
        info = {}
        return np.array(state_next, dtype=np.int32), reward, terminated, truncated, info