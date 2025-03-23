import random
import simpy
import numpy as np
import math
import streamlit as st
import scipy.stats
from dataclasses import dataclass
from typing import List, Tuple


def calculate_confidence_interval(data, confidence=0.95):
    a = np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


@dataclass
class SimulationParameters:
    num_reps: int
    random_seed: int
    num_checkin: int
    checkin_time: float
    patient_inter: float
    sim_time: int
    num_vaccinators: int
    vaccination_time: float
    num_adversewait: int
    adversewait_time: float


@dataclass
class SimulationResults:
    checkin_wait_time: float
    checkin_wait_num: float
    vaccination_wait_time: float
    vaccination_wait_num: float
    adverse_wait_time: float
    adverse_wait_num: float
    total_facility_time: float
    total_facility_time_ci: float
    num_vaccinated: float
    num_vaccinated_ci: float


class VaccinationClinic:
    def __init__(self, env, num_checkin, checkin_time, num_vaccinators, vaccination_time, num_adversewait, adversewait_time):
        self.env = env
        self.checkin_personnel = simpy.Resource(env, num_checkin)
        self.checkin_time = checkin_time
        self.vaccination_booth = simpy.Resource(env, num_vaccinators)
        self.vaccination_time = vaccination_time
        self.adverse_event_spot = simpy.Resource(env, num_adversewait)
        self.adversewait_time = adversewait_time

    def checkin(self, patient):
        yield self.env.timeout(np.random.triangular(
            max(0.2, self.checkin_time - 1), 
            self.checkin_time, 
            self.checkin_time + 1
        ))

    def vaccinate(self, patient):
        yield self.env.timeout(np.random.triangular(
            max(self.vaccination_time - 1, 0.2), 
            self.vaccination_time, 
            self.vaccination_time + 1
        ))

    def monitor_adverse(self, patient):
        yield self.env.timeout(self.adversewait_time)


class ClinicSimulation:
    def __init__(self, params: SimulationParameters):
        self.params = params
        random.seed(params.random_seed)
        
    def patient_process(self, env, name, clinic, times):
        arrival_time = env.now
        times.facility_arrivals.append(arrival_time)
        
        with clinic.checkin_personnel.request() as request:
            yield request
            times.checkin_begins.append(env.now)
            yield env.process(clinic.checkin(name))
            times.checkin_ends.append(env.now)

        with clinic.vaccination_booth.request() as request:
            yield request
            times.vaccination_begins.append(env.now)
            yield env.process(clinic.vaccinate(name))
            times.vaccination_ends.append(env.now)

        with clinic.adverse_event_spot.request() as request:
            yield request
            times.adverse_begins.append(env.now)
            yield env.process(clinic.monitor_adverse(name))
            times.adverse_ends.append(env.now)
            times.facility_departures.append(env.now)
    
    def setup(self, env, clinic, times):
        patient_count = 0
        while True:
            yield env.timeout(np.random.exponential(scale=self.params.patient_inter))
            patient_count += 1
            env.process(self.patient_process(env, f'Patient {patient_count}', clinic, times))

    def calculate_statistics(self, times):
        if not times.facility_departures:
            return {
                'total_time': 0,
                'checkin_wait_time': 0,
                'vaccination_wait_time': 0,
                'adverse_wait_time': 0,
                'avg_waiting_checkin': 0,
                'avg_waiting_vaccine': 0,
                'avg_waiting_adverse': 0,
                'num_vaccinated': 0
            }
            
        total_time = np.mean([
            times.facility_departures[i] - times.facility_arrivals[i] 
            for i in range(len(times.facility_departures))
        ])
        
        checkin_wait = np.mean([
            times.checkin_begins[i] - times.facility_arrivals[i] 
            for i in range(len(times.checkin_begins))
        ])
        
        vaccination_wait = np.mean([
            times.vaccination_begins[i] - times.checkin_ends[i] 
            for i in range(len(times.vaccination_begins))
        ])
        
        adverse_wait = np.mean([
            times.adverse_begins[i] - times.vaccination_ends[i] 
            for i in range(len(times.adverse_begins))
        ])
        
        avg_waiting_checkin = self.calculate_average_waiting(
            times.facility_arrivals, times.checkin_begins, self.params.sim_time
        )
        
        avg_waiting_vaccine = self.calculate_average_waiting(
            times.checkin_ends, times.vaccination_begins, self.params.sim_time
        )
        
        avg_waiting_adverse = self.calculate_average_waiting(
            times.vaccination_ends, times.adverse_begins, self.params.sim_time
        )
        
        return {
            'total_time': total_time,
            'checkin_wait_time': checkin_wait,
            'vaccination_wait_time': vaccination_wait,
            'adverse_wait_time': adverse_wait,
            'avg_waiting_checkin': avg_waiting_checkin,
            'avg_waiting_vaccine': avg_waiting_vaccine,
            'avg_waiting_adverse': avg_waiting_adverse,
            'num_vaccinated': len(times.facility_departures)
        }
    
    def calculate_average_waiting(self, end_times, begin_times, sim_time, interval=5):
        sample_points = range(0, int(sim_time), interval)
        waiting_counts = []
        
        for t in sample_points:
            num_ended = sum(1 for j in end_times if j <= t)
            num_began = sum(1 for j in begin_times if j <= t)
            waiting_counts.append(num_ended - num_began)
            
        return np.mean(waiting_counts) if waiting_counts else 0
        
    def run_single_replication(self):
        class SimulationTimes:
            def __init__(self):
                self.facility_arrivals = []
                self.checkin_begins = []
                self.checkin_ends = []
                self.vaccination_begins = []
                self.vaccination_ends = []
                self.adverse_begins = []
                self.adverse_ends = []
                self.facility_departures = []
        
        times = SimulationTimes()
        env = simpy.Environment()
        
        clinic = VaccinationClinic(
            env, 
            self.params.num_checkin,
            self.params.checkin_time,
            self.params.num_vaccinators,
            self.params.vaccination_time,
            self.params.num_adversewait,
            self.params.adversewait_time
        )
        
        env.process(self.setup(env, clinic, times))
        env.run(until=self.params.sim_time)
        
        return self.calculate_statistics(times)
    
    def run_simulation(self):
        progress_bar = st.progress(0.0)
        
        metrics = {
            'total_time': [],
            'checkin_wait_time': [],
            'vaccination_wait_time': [],
            'adverse_wait_time': [],
            'avg_waiting_checkin': [],
            'avg_waiting_vaccine': [],
            'avg_waiting_adverse': [],
            'num_vaccinated': []
        }
        
        for rep in range(self.params.num_reps):
            stats = self.run_single_replication()
            
            for key in metrics:
                metrics[key].append(stats[key])
                
            progress_bar.progress((rep + 1) / self.params.num_reps)
        
        progress_bar.progress(1.0)
        
        return SimulationResults(
            checkin_wait_time=np.mean(metrics['checkin_wait_time']),
            checkin_wait_num=np.mean(metrics['avg_waiting_checkin']),
            vaccination_wait_time=np.mean(metrics['vaccination_wait_time']),
            vaccination_wait_num=np.mean(metrics['avg_waiting_vaccine']),
            adverse_wait_time=np.mean(metrics['adverse_wait_time']),
            adverse_wait_num=np.mean(metrics['avg_waiting_adverse']),
            total_facility_time=np.mean(metrics['total_time']),
            total_facility_time_ci=calculate_confidence_interval(metrics['total_time']),
            num_vaccinated=np.mean(metrics['num_vaccinated']),
            num_vaccinated_ci=calculate_confidence_interval(metrics['num_vaccinated'])
        )


def render_ui():
    st.title('Vaccine Clinic Scheduling & Staffing Calculator')

    st.markdown("""
    This calculator allows you to experiment with patient scheduling and personnel staffing at a single vaccination clinic 
    to estimate the effects on desired operational goals and metrics. This calculator was developed as a collaboration between 
    [Dr. Sidd Nambiar](https://www.medicalhumanfactors.net/about-us/our-team/sidd-nambiar-phd/), [Dr. Sreenath Chalil Madathil](https://expertise.utep.edu/profiles/schalil), 
    and [Mr. Vishesh Kumar](http://visheshk.github.io/).
    
    The flow of patients through the clinic is assumed to be the following: Patients arrive to the facility according to a schedule. 
    Patients proceed to one (of maybe several) check-in stations. If all stations are occupied, patients wait in line.
    Following check-in, patients proceed to one of several available vaccination booths (or wait in line if all are busy).
    After getting a vaccine, patients are asked to proceed to a waiting area for approximately 15 minutes while they are monitored for
    adverse reactions. After 15 minutes, patients may safely leave the facility.
    
    If you would like to experiment with additional parameters or would like modifications, please feel free to reach out to Dr. Nambiar.
    
    Some technical notes: Patient arrivals are assumed to adhere to a poisson arrival process. Times to check-in and get a shot are assumed to be triangular around the mean. To play around with modifying these distributions, 
    please feel free to reach out.
    """)

    st.sidebar.title("Input values here")

    num_arrive_hour = st.sidebar.number_input("Patients expected per hour", min_value=1, value=30)
    num_waiting_area_adverse = st.sidebar.number_input("Waiting spots to monitor patients for adverse reactions", min_value=1, value=5)
    hours_facility_open = st.sidebar.number_input("Hours of facility opening", min_value=1, value=8)

    with st.sidebar.expander("Check-in counter parameters", True):
        checkin_time = st.number_input("Minutes for a single patient check-in", min_value=0.1, value=1.0)
        num_checkin = st.number_input("Input the number of check-in counters available for your patients", min_value=1, value=1)

    with st.sidebar.expander("Vaccination booth parameters", True):
        num_vaccine_booths = st.number_input("Vaccination booths", min_value=1, value=5)
        vaccination_time = st.number_input("Minutes for a single vaccination", min_value=0.1, value=4.0)    

    st.sidebar.write("\n   \n    \n")

    return {
        'num_arrive_hour': num_arrive_hour,
        'num_waiting_area_adverse': num_waiting_area_adverse,
        'hours_facility_open': hours_facility_open,
        'checkin_time': checkin_time,
        'num_checkin': num_checkin,
        'num_vaccine_booths': num_vaccine_booths,
        'vaccination_time': vaccination_time,
        'run_sim': st.sidebar.button('Calculate Metrics')
    }


def display_results(results: SimulationResults, hours_facility_open: int):
    if results.total_facility_time <= 30:
        st.success(f"Patients can expect to be in the facility for approximately {results.total_facility_time:0.1f} mins.")
    elif results.total_facility_time <= 60:
        st.warning(f"Patients can expect to be in the facility for approximately {results.total_facility_time:0.1f} mins.")
    else:
        st.error(f"Patients can expect to be in the facility for approximately {results.total_facility_time:0.1f} mins.")
        
    if results.checkin_wait_num <= 5:
        st.success(f"An average of {results.checkin_wait_num:0.0f} patients will wait in line for check-in")
    elif results.checkin_wait_num <= 15:
        st.warning(f"An average of {results.checkin_wait_num:0.0f} patients will wait in line for check-in. May need more check-in counters")
    else:
        st.error(f"An average of {results.checkin_wait_num:0.0f} patients will wait in line for check-in. Please add more check-in counters")

    if results.vaccination_wait_num < 5:
        st.success(f"An average of {results.vaccination_wait_num:0.0f} patients will wait in line between check-in and vaccination.")
    else:
        st.error(f"An average of {results.vaccination_wait_num:0.0f} patients will wait in line between check-in and vaccination. Please add more vaccination booths.")

    if results.adverse_wait_num <= 2:
        st.success(f"An average of {results.adverse_wait_num:0.0f} patients will not have adverse waiting spots.")
    else:
        st.error(f"An average of {results.adverse_wait_num:0.0f} patients will not have adverse waiting spots. Please add more.")
    
    st.info(f"Approximately {results.num_vaccinated:0.0f} patients can expect to be vaccinated during this {hours_facility_open} hour time-frame")


def main():
    inputs = render_ui()
    
    if inputs['run_sim']:
        random_seed = 42
        patient_inter = 60 / inputs['num_arrive_hour']
        num_reps = math.ceil(-0.114 * inputs['num_arrive_hour'] + 33.4)
        sim_time = 60 * inputs['hours_facility_open']
        adverse_wait_time = 15
        
        params = SimulationParameters(
            num_reps=num_reps,
            random_seed=random_seed,
            num_checkin=inputs['num_checkin'],
            checkin_time=inputs['checkin_time'],
            patient_inter=patient_inter,
            sim_time=sim_time,
            num_vaccinators=inputs['num_vaccine_booths'],
            vaccination_time=inputs['vaccination_time'],
            num_adversewait=inputs['num_waiting_area_adverse'],
            adversewait_time=adverse_wait_time
        )
        
        simulation = ClinicSimulation(params)
        results = simulation.run_simulation()
        
        display_results(results, inputs['hours_facility_open'])


if __name__ == "__main__":
    main()