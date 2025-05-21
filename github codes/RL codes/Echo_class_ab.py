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
    def __init__(self, time_start, time_close, rate_absence, render_env):
        self.time_start = time_start
        self.time_close = time_close
        self.rate_absence = rate_absence
        self.render_env = render_env
        self.state = {
            'patients': [],
        }

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
                                               'waiting time': 0,
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
                                                   'waiting time': 0,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'subspecialty': 'na'})
                    num += 1
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'waiting time': 0,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'subspecialty': 'na'})
                    num += 1
                # Otherwise there is one appointment scheduled
                else:
                    self.state['patients'].append({'patient number': num,
                                                   'status': 'incoming',
                                                   'Schedule time': time,
                                                   'waiting time': 0,
                                                   'Arrival time': 'NA',
                                                   'fetal': 'no',
                                                   'subspecialty': 'na'})
                    num += 1
            # Handle specific appointments made at different times
            if time in [self.convert_to_step('10:45'), self.convert_to_step('11:45')]:
                self.state['patients'].append({'patient number': num,
                                               'status': 'incoming',
                                               'Schedule time': time,
                                               'waiting time': 0,
                                               'Arrival time': 'NA',
                                               'fetal': 'no',
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
    # simulate the spell time of the echo test
    def _spell_time(self, patient_num):
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
        for i in range(action[0]):
            fetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'yes']
            max_waiting_patient = max(fetal_patients,
                                      key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x[
                                          'Schedule time'] else (self.env.now - x['Schedule time']))
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            self.env.process(self._spell_time(patient_num))
            self.state['patients'][patient_num]['waiting time'] = self.env.now - self.state['patients'][patient_num][
                'Arrival time']
            patients_ontime.remove(max_waiting_patient)
            fetal_patients.remove(max_waiting_patient)
        for i in range(action[1]):
            fetal_patients = [patient for patient in patients_late if patient['fetal'] == 'yes']
            max_waiting_patient = max(fetal_patients,
                                      key=lambda x: self.env.now - x['Arrival time'])
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'

            self.env.process(self._spell_time(patient_num))
            self.state['patients'][patient_num]['waiting time'] = self.env.now - self.state['patients'][patient_num][
                'Arrival time']
            patients_late.remove(max_waiting_patient)
            fetal_patients.remove(max_waiting_patient)
        for i in range(action[2]):
            fetal_patients = [patient for patient in patients_early if patient['fetal'] == 'yes']
            max_waiting_patient = max(fetal_patients,
                                      key=lambda x: self.env.now - x['Schedule time'])
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            self.env.process(self._spell_time(patient_num))
            self.state['patients'][patient_num]['waiting time'] = self.env.now - self.state['patients'][patient_num][
                'Arrival time']
            patients_early.remove(max_waiting_patient)
            fetal_patients.remove(max_waiting_patient)
        for i in range(action[3]):
            nonfetal_patients = [patient for patient in patients_ontime if patient['fetal'] == 'no']
            max_waiting_patient = max(nonfetal_patients,
                                      key=lambda x: (self.env.now - x['Arrival time']) if self.env.now > x[
                                          'Schedule time'] else (self.env.now - x['Schedule time']))
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            nonfetal_patients.remove(max_waiting_patient)
            patients_ontime.remove(max_waiting_patient)
            self.env.process(self._spell_time(patient_num))
            self.state['patients'][patient_num]['waiting time'] = self.env.now - self.state['patients'][patient_num][
                'Arrival time']
        for i in range(action[4]):
            nonfetal_patients = [patient for patient in patients_late if patient['fetal'] == 'no']
            max_waiting_patient = max(nonfetal_patients,
                                      key=lambda x: self.env.now - x['Arrival time'])
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            nonfetal_patients.remove(max_waiting_patient)
            patients_late.remove(max_waiting_patient)
            self.env.process(self._spell_time(patient_num))
            self.state['patients'][patient_num]['waiting time'] = self.env.now - self.state['patients'][patient_num][
                'Arrival time']
        for i in range(action[5]):
            nonfetal_patients = [patient for patient in patients_early if patient['fetal'] == 'no']
            max_waiting_patient = max(nonfetal_patients,
                                      key=lambda x: self.env.now - x['Schedule time'])
            patient_num = max_waiting_patient['patient number']
            max_waiting_patient['status'] = 'testing'
            nonfetal_patients.remove(max_waiting_patient)
            patients_early.remove(max_waiting_patient)
            self.env.process(self._spell_time(patient_num))
            self.state['patients'][patient_num]['waiting time'] = self.env.now - self.state['patients'][patient_num][
                'Arrival time']

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
              f"ontime waiting patients: {ontime_waiting_patients}, \n\n"
              f"late waiting patients: {late_waiting_patients}, \n\n"
              f"early waiting patients: {early_waiting_patients}, \n\n"
              f"fetal waiting patients: {fetal_waiting_patients}, \n\n"
              f"nonfetal waiting patients: {nonfetal_waiting_patients}, \n\n"
              )

        # the function to reset
    def reset(self, seed=None, options=None):
        """Reset the environment."""

        # Handle the seed for reproducibility
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            # Also set the seed for other random libraries if necessary
            random.seed(seed)

        # Reset the environment state
        self.env = simpy.Environment()
        # Reapply all parameters from initialization
        self.state = {
            'patients': [],
        }
        self._load_schedules()
        self._load_patients()
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
            self.env.now
        ]

        # Return the initial observation and an empty info dictionary
        return np.array(state, dtype=np.int32), {}


    # penalty calculation
    def _penalty_calculation(self):
        penalty = 0
        patients_waiting_ontime = []
        patients_waiting_late = []
        patients_waiting_early = []
        for patient in self.state['patients']:
            if patient['status'] == 'waiting(ontime)':
                patients_waiting_ontime.append(patient)
            elif patient['status'] == 'waiting(late)':
                patients_waiting_late.append(patient)
            #            elif patient['status'] == 'waiting(early)':
            #                patients_waiting_early.append(patient)
        if self.env.now >= self.convert_to_step(self.time_close):
            for patient in self.state['patients']:
                if 'waiting' in patient['status']:
                    penalty += 10
        for patient in patients_waiting_ontime:
                #            if self.env.now >= patient['Schedule time']:
            penalty += 4
            #                penalty += 2 * (self.env.now - patient['Arrival time'])
        for patient in patients_waiting_late:
            penalty += 2
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
            self.env.now
        ]

        # Return the new state, reward, termination flag, truncated flag, and any additional info
        info = {}
        return np.array(state_next, dtype=np.int32), reward, terminated, truncated, info