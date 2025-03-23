import random
import simpy
import numpy as np
import math
import streamlit as st
import scipy.stats
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> float:
    """
    Calculates the confidence interval for a given dataset.

    Args:
        data (List[float]): The list of data points.
        confidence (float): The desired confidence level (default: 0.95).

    Returns:
        float: The margin of error for the confidence interval.
    """
    try:
        a = np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return h
    except Exception as e:
        logging.error(f"Error calculating confidence interval: {e}")
        return 0.0


@dataclass
class SimulationParameters:
    """
    Data class to hold simulation parameters.  Using default values for easier UI integration.
    """
    num_reps: int = 30  # Reduced default reps, can be increased if needed.
    random_seed: int = 42
    num_checkin: int = 1
    checkin_time: float = 1.0
    patient_inter: float = 2.0
    sim_time: int = 480  # 8 hours in minutes
    num_vaccinators: int = 5
    vaccination_time: float = 4.0
    num_adversewait: int = 5
    adversewait_time: float = 15.0
    arrival_distribution: str = "exponential"  # Add arrival distribution choice

@dataclass
class SimulationResults:
    """
    Data class to hold simulation results. Using default values.
    """
    checkin_wait_time: float = 0.0
    checkin_wait_num: float = 0.0
    vaccination_wait_time: float = 0.0
    vaccination_wait_num: float = 0.0
    adverse_wait_time: float = 0.0
    adverse_wait_num: float = 0.0
    total_facility_time: float = 0.0
    total_facility_time_ci: float = 0.0
    num_vaccinated: float = 0.0
    num_vaccinated_ci: float = 0.0
    utilization_checkin: float = 0.0
    utilization_vaccination: float = 0.0
    utilization_adverse: float = 0.0

class VaccinationClinic:
    """
    Represents the vaccination clinic environment.
    """

    def __init__(self, env: simpy.Environment, params: SimulationParameters):
        """
        Initializes the VaccinationClinic.

        Args:
            env (simpy.Environment): The SimPy environment.
            params (SimulationParameters): The simulation parameters.
        """
        self.env = env
        self.params = params
        self.checkin_personnel = simpy.Resource(env, capacity=params.num_checkin)
        self.vaccination_booth = simpy.Resource(env, capacity=params.num_vaccinators)
        self.adverse_event_spot = simpy.Resource(env, capacity=params.num_adversewait)

        # Monitor resource utilization
        self.checkin_usage = []
        self.vaccination_usage = []
        self.adverse_usage = []

    def checkin(self, patient):
        """Simulates the check-in process."""
        yield self.env.timeout(np.random.triangular(
            max(0.2, self.params.checkin_time - 1),
            self.params.checkin_time,
            self.params.checkin_time + 1
        ))

    def vaccinate(self, patient):
        """Simulates the vaccination process."""
        yield self.env.timeout(np.random.triangular(
            max(self.params.vaccination_time - 1, 0.2),
            self.params.vaccination_time,
            self.params.vaccination_time + 1
        ))

    def monitor_adverse(self, patient):
        """Simulates monitoring for adverse events."""
        yield self.env.timeout(self.params.adversewait_time)


class ClinicSimulation:
    """
    Orchestrates the simulation of the vaccination clinic.
    """

    def __init__(self, params: SimulationParameters):
        """
        Initializes the ClinicSimulation.

        Args:
            params (SimulationParameters): The simulation parameters.
        """
        self.params = params
        random.seed(params.random_seed)

    def patient_process(self, env: simpy.Environment, name: str, clinic: VaccinationClinic, times: Dict[str, list]):
        """
        Simulates a single patient's journey through the clinic.

        Args:
            env (simpy.Environment): The SimPy environment.
            name (str): The name of the patient.
            clinic (VaccinationClinic): The vaccination clinic object.
            times (Dict[str, list]): A dictionary to store timestamps of events.
        """
        arrival_time = env.now
        times['facility_arrivals'].append(arrival_time)

        with clinic.checkin_personnel.request() as request:
            yield request
            times['checkin_begins'].append(env.now)
            yield env.process(clinic.checkin(name))
            times['checkin_ends'].append(env.now)
            clinic.checkin_usage.append((env.now, len(clinic.checkin_personnel.queue)))

        with clinic.vaccination_booth.request() as request:
            yield request
            times['vaccination_begins'].append(env.now)
            yield env.process(clinic.vaccinate(name))
            times['vaccination_ends'].append(env.now)
            clinic.vaccination_usage.append((env.now, len(clinic.vaccination_booth.queue)))

        with clinic.adverse_event_spot.request() as request:
            yield request
            times['adverse_begins'].append(env.now)
            yield env.process(clinic.monitor_adverse(name))
            times['adverse_ends'].append(env.now)
            times['facility_departures'].append(env.now)
            clinic.adverse_usage.append((env.now, len(clinic.adverse_event_spot.queue)))

    def setup(self, env: simpy.Environment, clinic: VaccinationClinic, times: Dict[str, list]):
        """
        Sets up the simulation by creating patients at specified intervals.

        Args:
            env (simpy.Environment): The SimPy environment.
            clinic (VaccinationClinic): The vaccination clinic object.
            times (Dict[str, list]): A dictionary to store timestamps of events.
        """
        patient_count = 0
        while True:
            if self.params.arrival_distribution == "exponential":
                interarrival_time = np.random.exponential(scale=self.params.patient_inter)
            elif self.params.arrival_distribution == "fixed":
                interarrival_time = self.params.patient_inter
            else:
                raise ValueError(f"Unknown arrival distribution: {self.params.arrival_distribution}")

            yield env.timeout(interarrival_time)
            patient_count += 1
            env.process(self.patient_process(env, f'Patient {patient_count}', clinic, times))

    def calculate_statistics(self, times: Dict[str, list], clinic: VaccinationClinic) -> Dict[str, float]:
        """
        Calculates key performance indicators (KPIs) from the simulation.

        Args:
            times (Dict[str, list]): A dictionary containing timestamps of events.
            clinic (VaccinationClinic): The vaccination clinic object.

        Returns:
            Dict[str, float]: A dictionary containing the calculated statistics.
        """
        num_departures = len(times['facility_departures'])

        if not num_departures:
            return {
                'total_time': 0,
                'checkin_wait_time': 0,
                'vaccination_wait_time': 0,
                'adverse_wait_time': 0,
                'avg_waiting_checkin': 0,
                'avg_waiting_vaccine': 0,
                'avg_waiting_adverse': 0,
                'num_vaccinated': 0,
                'utilization_checkin': 0,
                'utilization_vaccination': 0,
                'utilization_adverse': 0
            }

        total_time = np.mean([
            times['facility_departures'][i] - times['facility_arrivals'][i]
            for i in range(num_departures)
        ])

        checkin_wait = np.mean([
            times['checkin_begins'][i] - times['facility_arrivals'][i]
            for i in range(len(times['checkin_begins']))
        ])

        vaccination_wait = np.mean([
            times['vaccination_begins'][i] - times['checkin_ends'][i]
            for i in range(len(times['vaccination_begins']))
        ])

        adverse_wait = np.mean([
            times['adverse_begins'][i] - times['vaccination_ends'][i]
            for i in range(len(times['adverse_begins']))
        ])

        avg_waiting_checkin = self.calculate_average_waiting(
            times['facility_arrivals'], times['checkin_begins'], self.params.sim_time
        )

        avg_waiting_vaccine = self.calculate_average_waiting(
            times['checkin_ends'], times['vaccination_begins'], self.params.sim_time
        )

        avg_waiting_adverse = self.calculate_average_waiting(
            times['vaccination_ends'], times['adverse_begins'], self.params.sim_time
        )

        utilization_checkin = self.calculate_resource_utilization(clinic.checkin_usage, self.params.sim_time, clinic.checkin_personnel.capacity)
        utilization_vaccination = self.calculate_resource_utilization(clinic.vaccination_usage, self.params.sim_time, clinic.vaccination_booth.capacity)
        utilization_adverse = self.calculate_resource_utilization(clinic.adverse_usage, self.params.sim_time, clinic.adverse_event_spot.capacity)

        return {
            'total_time': total_time,
            'checkin_wait_time': checkin_wait,
            'vaccination_wait_time': vaccination_wait,
            'adverse_wait_time': adverse_wait,
            'avg_waiting_checkin': avg_waiting_checkin,
            'avg_waiting_vaccine': avg_waiting_vaccine,
            'avg_waiting_adverse': avg_waiting_adverse,
            'num_vaccinated': num_departures,
            'utilization_checkin': utilization_checkin,
            'utilization_vaccination': utilization_vaccination,
            'utilization_adverse': utilization_adverse
        }

    def calculate_average_waiting(self, end_times: List[float], begin_times: List[float], sim_time: int, interval: int = 5) -> float:
        """
        Calculates the average number of patients waiting in a queue over time.

        Args:
            end_times (List[float]): List of times when patients finished the previous stage.
            begin_times (List[float]): List of times when patients started the current stage.
            sim_time (int): Total simulation time.
            interval (int): Sampling interval for calculating the average.

        Returns:
            float: The average number of patients waiting.
        """
        sample_points = range(0, int(sim_time), interval)
        waiting_counts = []

        for t in sample_points:
            num_ended = sum(1 for j in end_times if j <= t)
            num_began = sum(1 for j in begin_times if j <= t)
            waiting_counts.append(num_ended - num_began)

        return np.mean(waiting_counts) if waiting_counts else 0

    def calculate_resource_utilization(self, usage_data, sim_time, capacity):
        """Calculates the average utilization of a resource over the simulation time."""
        if not usage_data:
            return 0.0

        total_busy_time = 0
        previous_time = 0

        for time, queue_length in usage_data:
            # utilization = queue_length / capacity  # calculate the utilization as the queue length relative to capacity
            utilization = min(queue_length, capacity) / capacity  # utilization should not be over 1
            total_busy_time += utilization * (time - previous_time)
            previous_time = time

        return (total_busy_time / sim_time) if sim_time > 0 else 0

    def run_single_replication(self) -> Dict[str, float]:
        """
        Runs a single replication of the simulation.

        Returns:
            Dict[str, float]: A dictionary containing the statistics for this replication.
        """
        times = {
            'facility_arrivals': [],
            'checkin_begins': [],
            'checkin_ends': [],
            'vaccination_begins': [],
            'vaccination_ends': [],
            'adverse_begins': [],
            'adverse_ends': [],
            'facility_departures': []
        }

        env = simpy.Environment()
        clinic = VaccinationClinic(env, self.params)
        env.process(self.setup(env, clinic, times))
        env.run(until=self.params.sim_time)

        return self.calculate_statistics(times, clinic)

    def run_simulation(self) -> SimulationResults:
        """
        Runs multiple replications of the simulation and aggregates the results.

        Returns:
            SimulationResults: An object containing the aggregated simulation results.
        """
        progress_bar = st.progress(0.0)

        metrics = {
            'total_time': [],
            'checkin_wait_time': [],
            'vaccination_wait_time': [],
            'adverse_wait_time': [],
            'avg_waiting_checkin': [],
            'avg_waiting_vaccine': [],
            'avg_waiting_adverse': [],
            'num_vaccinated': [],
            'utilization_checkin': [],
            'utilization_vaccination': [],
            'utilization_adverse': []
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
            num_vaccinated_ci=calculate_confidence_interval(metrics['num_vaccinated']),
            utilization_checkin=np.mean(metrics['utilization_checkin']),
            utilization_vaccination=np.mean(metrics['utilization_vaccination']),
            utilization_adverse=np.mean(metrics['utilization_adverse'])
        )


def render_ui() -> Dict:
    """
    Renders the Streamlit user interface.

    Returns:
        Dict: A dictionary containing the input values from the UI.
    """
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

    st.sidebar.title("Input Values")

    num_arrive_hour = st.sidebar.number_input("Patients per hour", min_value=1, value=30, help="Expected number of patients arriving per hour.")
    num_waiting_area_adverse = st.sidebar.number_input("Adverse Monitoring Spots", min_value=1, value=5, help="Number of waiting spots for monitoring adverse reactions.")
    hours_facility_open = st.sidebar.number_input("Facility Open (Hours)", min_value=1, value=8, help="Number of hours the facility is open.")

    with st.sidebar.expander("Check-in Configuration", True):
        checkin_time = st.number_input("Check-in Time (Minutes)", min_value=0.1, value=1.0, help="Average time for a single patient check-in.")
        num_checkin = st.number_input("Check-in Counters", min_value=1, value=1, help="Number of check-in counters available.")

    with st.sidebar.expander("Vaccination Booth Configuration", True):
        num_vaccine_booths = st.number_input("Vaccination Booths", min_value=1, value=5, help="Number of vaccination booths.")
        vaccination_time = st.number_input("Vaccination Time (Minutes)", min_value=0.1, value=4.0, help="Average time for a single vaccination.")

    arrival_distribution = st.sidebar.selectbox(
        "Arrival Distribution",
        options=["exponential", "fixed"],
        index=0,  # Default to exponential
        help="Choose the distribution for patient arrivals. Exponential is a common assumption for healthcare settings.  Fixed provides regularly spaced arrivals for comparison."
    )

    st.sidebar.write("\n   \n    \n")  # Add some space

    run_sim = st.sidebar.button('Run Simulation', help="Click to start the simulation with the current parameters.")

    return {
        'num_arrive_hour': num_arrive_hour,
        'num_waiting_area_adverse': num_waiting_area_adverse,
        'hours_facility_open': hours_facility_open,
        'checkin_time': checkin_time,
        'num_checkin': num_checkin,
        'num_vaccine_booths': num_vaccine_booths,
        'vaccination_time': vaccination_time,
        'arrival_distribution': arrival_distribution,
        'run_sim': run_sim
    }


def display_results(results: SimulationResults, hours_facility_open: int):
    """
    Displays the simulation results in a user-friendly format.

    Args:
        results (SimulationResults): The simulation results object.
        hours_facility_open (int): The number of hours the facility is open.
    """
    st.header("Simulation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Experience")
        display_time_result("Total Time in Facility", results.total_facility_time)
        display_wait_result("Check-in Wait", results.checkin_wait_num)
        display_wait_result("Vaccination Wait", results.vaccination_wait_num)
        display_wait_result("Adverse Monitoring Wait", results.adverse_wait_num)

    with col2:
        st.subheader("Operational Metrics")
        st.metric("Patients Vaccinated", f"{results.num_vaccinated:0.0f}")
        st.metric("Check-in Utilization", f"{results.utilization_checkin:.2f}")
        st.metric("Vaccination Booth Utilization", f"{results.utilization_vaccination:.2f}")
        st.metric("Adverse Spot Utilization", f"{results.utilization_adverse:.2f}")


def display_time_result(metric_name: str, time_value: float):
    """Displays a time-related result with appropriate coloring."""
    if time_value <= 30:
        st.success(f"{metric_name}: {time_value:0.1f} mins")
    elif time_value <= 60:
        st.warning(f"{metric_name}: {time_value:0.1f} mins")
    else:
        st.error(f"{metric_name}: {time_value:0.1f} mins")


def display_wait_result(metric_name: str, wait_value: float):
    """Displays a waiting-related result with appropriate coloring."""
    if wait_value <= 2:
        st.success(f"{metric_name}: {wait_value:0.0f} patients")
    elif wait_value <= 5:
        st.warning(f"{metric_name}: {wait_value:0.0f} patients")
    else:
        st.error(f"{metric_name}: {wait_value:0.0f} patients")


def main():
    """
    Main function to run the Streamlit application.
    """
    inputs = render_ui()

    if inputs['run_sim']:
        try:
            # Parameter validation
            if not all(isinstance(value, (int, float)) and value > 0 for key, value in inputs.items() if key not in ['run_sim', 'arrival_distribution']):
                st.error("Please ensure all input values are positive numbers.")
                return

            patient_inter = 60 / inputs['num_arrive_hour']
            num_reps = math.ceil(-0.114 * inputs['num_arrive_hour'] + 33.4)  # Consider a minimum number of reps
            num_reps = max(num_reps, 10) #Setting a minimum number of reps
            sim_time = 60 * inputs['hours_facility_open']
            adverse_wait_time = 15

            params = SimulationParameters(
                num_reps=num_reps,
                random_seed=42,
                num_checkin=inputs['num_checkin'],
                checkin_time=inputs['checkin_time'],
                patient_inter=patient_inter,
                sim_time=sim_time,
                num_vaccinators=inputs['num_vaccine_booths'],
                vaccination_time=inputs['vaccination_time'],
                num_adversewait=inputs['num_waiting_area_adverse'],
                adversewait_time=adverse_wait_time,
                arrival_distribution=inputs['arrival_distribution']
            )

            simulation = ClinicSimulation(params)
            results = simulation.run_simulation()

            display_results(results, inputs['hours_facility_open'])

        except Exception as e:
            logging.exception("An error occurred during the simulation:")  # Log the full exception including traceback
            st.error(f"An unexpected error occurred. Please check your inputs and try again.  Details: {e}")


if __name__ == "__main__":
    main()