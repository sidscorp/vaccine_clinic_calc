import random
import simpy
import numpy as np
import math
import streamlit as st

import scipy.stats



def conf_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h



def run_vaccination_simulation(NUM_REPS, RANDOM_SEED, NUM_CHECKIN, CHECKIN_TIME, PATIENT_INTER, SIM_TIME, NUM_VACCINATORS, VACCINATION_TIME, NUM_ADVERSEWAIT, ADVERSEWAIT_TIME):
    output_checkin_waittime = []
    output_checkin_waitnum = []
    output_vaccination_waittime = []
    output_vaccination_waitnum = []
    output_adverse_waittime = []
    output_adverse_waitnum = []
    output_total_facility_time = []
    output_num_vaccinated = []
    my_bar = st.progress(0.0)
    reps = math.ceil(NUM_REPS)
    for replication in range(0,reps):
        percent_complete = float(replication/reps)
        facility_arrival_times = []
        checkin_begin_times = []
        checkin_end_times = []
        vaccination_begin_times = []
        vaccination_end_times = []
        adverse_begin_times = []
        adverse_end_times = []
        facility_departure_times = []

        class Vaccination_Clinic(object):
            def __init__(self, env, num_checkin, checkin_time, num_vaccinators, vaccination_time, num_adversewait, adversewait_time):
                self.env = env
                self.checkin_personnel = simpy.Resource(env, num_checkin)
                self.checkintime = checkin_time
                self.vaccination_booth = simpy.Resource(env, num_vaccinators)
                self.vaccinationtime = vaccination_time
                self.adverse_event_spot = simpy.Resource(env, num_adversewait)
                self.adversewaittime = adversewait_time

            def checkin(self, patient):
                yield self.env.timeout(np.random.triangular(CHECKIN_TIME - 1, CHECKIN_TIME, CHECKIN_TIME + 1))


            def vaccinate(self, patient):
                yield self.env.timeout(np.random.triangular(VACCINATION_TIME - 1, VACCINATION_TIME, VACCINATION_TIME + 1))

            def monitor_adverse(self, patient):
                yield self.env.timeout(ADVERSEWAIT_TIME)

        def patient(env, name, vac):
            #print('%s arrives at the vaccination clinic at %.2f.' % (name, env.now))
            facility_arrival_times.append(env.now)
            with vac.checkin_personnel.request() as request:
                yield request

                #print("%s arrives at checkin counter" % name)
                checkin_begin_times.append(env.now)
                yield env.process(vac.checkin(name))
                checkin_end_times.append(env.now)
                #print("%s completes check-in at %.2f." % (name, env.now))

            with vac.vaccination_booth.request() as request:
                yield request
                vaccination_begin_times.append(env.now)
                #print("%s arrives at vaccination booth" % name)
                yield env.process(vac.vaccinate(name))
                vaccination_end_times.append(env.now)

                #print("%s gets shot in the arm at %.2f." % (name, env.now))

            with vac.adverse_event_spot.request() as request:
                yield request
                adverse_begin_times.append(env.now)
                #print("%s proceeds to wait to monitor for adverse events" % name)
                yield env.process(vac.monitor_adverse(name))
                adverse_end_times.append(env.now)
                facility_departure_times.append(env.now)
                #print("%s leaves facility safely at %.2f." % (name, env.now))

        def setup(env, num_checkin, checkin_time, num_vaccinators, vaccination_time, num_adversewait, adversewait_time, patient_inter):
            vaccinationclinic = Vaccination_Clinic(env, num_checkin, checkin_time, num_vaccinators, vaccination_time, num_adversewait, adversewait_time)
            i = 0
            while True:
                yield env.timeout(np.random.exponential(scale=patient_inter))
                i += 1
                env.process(patient(env, 'Patient %d' % i, vaccinationclinic))   


        random.seed(RANDOM_SEED)

        # Create an environment and start the setup process
        env = simpy.Environment()
        env.process(setup(env, NUM_CHECKIN, CHECKIN_TIME, NUM_VACCINATORS, VACCINATION_TIME, NUM_ADVERSEWAIT, ADVERSEWAIT_TIME, PATIENT_INTER))

        # Execute!
        env.run(until=SIM_TIME)
        average_facility_total_time = np.mean([facility_departure_times[i] - facility_arrival_times[i] for i in range(len(facility_departure_times))])
        #print("Approximate total time at facility is %.1f mins." % average_facility_total_time)
        average_checkin_wait_time = np.mean([checkin_begin_times[i] - facility_arrival_times[i] for i in range(len(checkin_begin_times))])
        #print("Approximate wait time between arrival and checkin is %.1f mins." % average_checkin_wait_time)
        average_vaccination_wait_time = np.mean([vaccination_begin_times[i] - checkin_end_times[i] for i in range(len(vaccination_begin_times))])
        #print("Approximate wait time between checkin and getting vaccinated is %.1f mins." % average_vaccination_wait_time)
        average_adverse_wait_time = np.mean([adverse_begin_times[i] - vaccination_end_times[i] for i in range(len(adverse_begin_times))])
        #print("Approximate wait time between getting vaccine and finding adverse monitoring wait spot is %.1f mins." % average_adverse_wait_time)

        avg_waiting_checkin = []
        for i in [x * 0.1 for x in range(0, SIM_TIME*10)]:
            num_arrived_facility = sum(1 for j in facility_arrival_times if j <= i)
            num_started_checin = sum(1 for j in checkin_begin_times if j <= i)
            avg_waiting_checkin.append(num_arrived_facility - num_started_checin)
        #print("Approximate # of patients waiting to checkin at any time is %.1f." % np.mean(avg_waiting_checkin))

        avg_waiting_vaccine = []
        for i in [x * 0.1 for x in range(0, SIM_TIME*10)]:
            num_finished_checin = sum(1 for j in checkin_end_times if j <= i)
            num_started_vaccine = sum(1 for j in vaccination_begin_times if j <= i)
            avg_waiting_vaccine.append(num_finished_checin - num_started_vaccine)
        #print("Approximate # of patients waiting between checkin and vaccine at any time is %.1f." % np.mean(avg_waiting_vaccine))

        avg_waiting_adverse = []
        for i in [x * 0.1 for x in range(0, SIM_TIME*10)]:
            num_finished_vaccine = sum(1 for j in vaccination_end_times if j <= i)
            num_started_adverse = sum(1 for j in adverse_begin_times if j <= i)
            avg_waiting_adverse.append(num_finished_vaccine - num_started_adverse)
        #print("Approximate # of patients waiting between checkin and vaccine at any time is %.1f." % np.mean(avg_waiting_vaccine))

        output_checkin_waittime.append(average_checkin_wait_time)
        output_checkin_waitnum.append(np.mean(avg_waiting_checkin))
        output_vaccination_waittime.append(average_vaccination_wait_time)
        output_vaccination_waitnum.append(np.mean(avg_waiting_vaccine))
        output_adverse_waittime.append(average_adverse_wait_time)
        output_adverse_waitnum.append(np.mean(avg_waiting_adverse))
        output_total_facility_time.append(average_facility_total_time)
        output_num_vaccinated.append(len(facility_departure_times))
        
        my_bar.progress(float(percent_complete))
    my_bar.progress(1.0)   
    return [np.mean(output_checkin_waittime), np.mean(output_checkin_waitnum), np.mean(output_vaccination_waittime), 
            np.mean(output_vaccination_waitnum), np.mean(output_adverse_waittime), np.mean(output_adverse_waitnum), conf_interval(output_total_facility_time),
            np.mean(output_total_facility_time), conf_interval(output_num_vaccinated), np.mean(output_num_vaccinated)]



st.title('Vaccine Clinic Scheduling & Staffing Calculator')

st.markdown('This calculator allows you to experiment with patient scheduling and personnel staffing at a single vaccination clinic \
            to estimate the effects on desired operational goals and metrics.')
st.markdown('The flow of patients through the clinic is assumed to be the following: Patients arrive to the facility according to a schedule. \
            Patients proceed to one (of maybe several) check-in stations. If all stations are occupied, patients wait in line.\
            Following check-in, patients proceed to one of several available vaccination booths (or wait in line if all are busy).\
            After getting a vaccine, patients are asked to proceed to a waiting area for approximately 15 minutes while they are monitored for\
            adverse reactions. After 15 minutes, patients may safely leave the facility.')
st.markdown('It is assumed that check-in takes approximately 1 minute and that the vaccination process takes approximately 4 minutes.\
            If you would like to experiment with these parameters as well, feel free to reach out to the developer (Sidd Nambiar; Twitter: @SiddNambiar).')
num_arrive_hour = st.number_input("Input the number of patients you expect will arrive in an hour", min_value = 1)
num_checkin = st.number_input("Input the number of check-in counters available for your patients", min_value = 1)
num_vaccine_booths = st.number_input("Input the number of vaccination booths available at your location", min_value = 1)
num_waiting_area_adverse = st.number_input("Input the number of waiting spots available for patients while being monitored for adverse reactions", min_value = 1)





if(st.button('Calculate Metrics')): 
    RANDOM_SEED = 42
    NUM_CHECKIN = num_checkin
    CHECKIN_TIME = 1
    PATIENT_INTER = 60/num_arrive_hour
    NUM_REPS = max(10, (num_arrive_hour*8)/500)
    SIM_TIME = 60*8
    NUM_VACCINATORS = num_vaccine_booths
    VACCINATION_TIME = 4
    NUM_ADVERSEWAIT = num_waiting_area_adverse
    ADVERSEWAIT_TIME = 15
    [avg_checkin_waitT, avg_checkin_waitN, avg_vaccine_waitT, avg_vaccine_waitN, avg_adverse_waitT, avg_adverse_waitN, conf_total_time, avg_total_time, conf_total_vaccinated, tot_num_vaccinated] = run_vaccination_simulation(NUM_REPS, RANDOM_SEED, NUM_CHECKIN, CHECKIN_TIME, PATIENT_INTER, SIM_TIME, NUM_VACCINATORS, VACCINATION_TIME, NUM_ADVERSEWAIT, ADVERSEWAIT_TIME)
    if(avg_total_time <= 30):
        st.success("Patients can expect to be in the facility for {:0.1f} (+/- {:0.1f}) mins.".format(avg_total_time, conf_total_time)) 
    elif(avg_total_time <= 60):
        st.warning("Patients can expect to be in the facility for {:0.1f} (+/- {:0.1f}) mins.".format(avg_total_time, conf_total_time))
    else:
        st.error("Patients can expect to be in the facility for {:0.1f} (+/- {:0.1f}) mins.".format(avg_total_time, conf_total_time))
    st.info("Approximately {:0.1f} (+/- {:0.1f}) patients can expect to be vaccinated during this 8 hour day".format(tot_num_vaccinated, conf_total_vaccinated))
    #st.info("Approximately {:0.1f} patients must wait before check-in".format(avg_checkin_waitN)) 
    #st.info("Patients can expect to wait for approximately {:0.1f} mins to check-in".format(avg_checkin_waitT))
    #st.info("Approximately {:0.1f} patients must wait between check-in and getting vaccine".format(avg_vaccine_waitN)) 
