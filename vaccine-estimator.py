# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 18:06:54 2021

@author: siddh
"""

import math
import streamlit as st

class MMcQueue(object):
    def __init__(self, arrival, departure, capacity):
        """
        Given the parameter of one M/M/c/c Queue, 
        initialize the queue with these parameter and calculate some parameters.
        `_rou`:     Server Utilization
        `_p0`:      Probability of that there is no packets in the queue
        `_pc`:      Probability of that there is exactly `capacity` packets in the queue,
                    that is, all the server is busy.
        `_probSum`:  p0 + p1 + p2 + ... pc - pc
        `_finalTerm`: 1/(c!) * (arrival / departure)^c
        """
        if capacity * departure <= arrival:
            raise ValueError("This Queue is unstable with the Input Parameters!!!")
        self._arrival = float(arrival)
        self._departure = float(departure)
        self._capacity = capacity
        self._rou = self._arrival / self._departure / self._capacity

        # init the parameter as if the capacity == 0
        powerTerm = 1.0
        factorTerm = 1.0
        preSum = 1.0
        # Loop through `1` to `self._capacity` to get each term and preSum
        for i in range(1, self._capacity + 1):
            powerTerm *= self._arrival / self._departure
            factorTerm /= i
            preSum += powerTerm * factorTerm
        self._finalTerm = powerTerm * factorTerm
        preSum -= self._finalTerm
        self._p0 = 1.0 / (preSum + self._finalTerm / (1 - self._rou))
        self._pc = self._finalTerm * self._p0
        self._probSum = preSum * self._p0


    @property
    def arrival(self):
        return self._arrival

    @property
    def departure(self):
        return self._departure

    @property
    def capacity(self):
        return self._capacity

    def getPk(self, k):
        """
        Return the probability when there are `k` packets in the system
        """
        if k == 0:
            return self._p0
        elif k == self._capacity:
            return self._pc
        elif k < self._capacity:
            factorTerm = 1.0 / math.factorial(k)
            powerTerm = math.pow(self._arrival / self._departure, k)
            return self._p0 * factorTerm * powerTerm
        else:
            return self._finalTerm * math.pow(self._rou, k - self._capacity) * self._p0

    def getQueueProb(self):
        """
        Return the probability when a packet comes, it needs to queue in the buffer.
        That is, P(W>0) = 1 - P(N < c)
        Also known as Erlang-C function
        """
        return 1.0 - self._probSum

    def getIdleProb(self):
        """
        Return the probability when the sever is idle.
        That is , P(N=0)
        """
        return self._p0

    def getAvgPackets(self):
        """
        Return the average number of packets in the system (in service and in the queue)
        """
        return self._rou / (1 - self._rou) * self.getQueueProb() + self._capacity * self._rou

    def getAvgQueueTime(self):
        """
        Return the average time of packets spending in the queue
        """
        return self.getQueueProb() / (self._capacity * self._departure - self._arrival)

    def getAvgQueuePacket_Given(self):
        """
        Given there is packet in the queue,
        return the average number of packets in the queue
        """
        return self._finalTerm * self._p0 / (1.0 - self._rou) / (1.0 - self._rou)

    def getAvgQueueTime_Given(self):
        """
        Given a packet must wait, 
        return the average time of this packet spending in the queue
        """
        if self.getQueueProb() == 0:
            return 0
        return self.getAvgQueuePacket_Given() / (self.getQueueProb() * self._arrival)

    def getAvgResponseTime(self):
        """
        Return the average time of packets spending in the system (in service and in the queue)
        """
        return self.getAvgQueueTime() + 1.0 / self._departure

    def getAvgPacketInSystem(self):
        """
        Return the average number of packets in the system.
        """
        return self.getAvgResponseTime() * self._arrival

    def getAvgBusyServer(self):
        """
        Return the average number of busy Server.
        """
        return self.arrival / self.departure


    def getPorbWhenQueueTimeLargerThan(self, queueTime):
        """
        Return the probability when the queuing time of the packet is larger than `queueTime`
        That is P(W > queueTime) = 1 - P(W <= queueTime)
        """
        firstTerm = self._pc / (1.0 - self._rou)
        expTerm = - self._capacity * self._departure * (1.0 - self._rou) * queueTime
        secondTerm = math.exp(expTerm)
        return firstTerm * secondTerm


def calc_metrics(lambd, mu, c):
    thisQueue = MMcQueue(lambd,mu,c)
    Wq = thisQueue.getAvgQueueTime()
    Lq = lambd*Wq
    server_busy = thisQueue.getAvgBusyServer()/c
    L = thisQueue.getAvgPacketInSystem()
    W = L/lambd
    return [L, W, Lq, Wq, server_busy]

st.title('Vaccine Clinic Scheduling & Staffing Calculator')

st.write("This calculator allows you to experiment with scheduling and staffing of patients arriving at a vaccination clinic")
num_arrive_hour = st.number_input("Input the number of patients you expect will arrive in an hour", min_value = 1)
num_checkin = st.number_input("Input the number of check-in counters available for your patients", min_value = 1)
num_vaccine_booths = st.number_input("Input the number of vaccination booths available at your location", min_value = 1)
num_waiting_area_adverse = st.number_input("Input the number of waiting spots available for patients while being monitored for adverse reactions", min_value = 1)





if(st.button('Calculate Metrics')): 
    lambd = num_arrive_hour/60
    mu_checkin = 1
    mu_vaccinate = 4
    if(lambd/(mu_checkin*num_checkin) >= 1):
        st.warning("Warning: Unstable queue. Increase number of check-in personnel.")
    elif(lambd/(mu_vaccinate*num_vaccine_booths) >= 1):
        st.warning("Warning: Unstable queue. Increase number of vaccination booths.")
    else:
        [L_checin, W_checkin, Lq_checkin, Wq_checkin, Busy_checkin] = calc_metrics(lambd, mu_checkin, num_checkin)
        [L_vaccine, W_vaccine, Lq_vaccine, Wq_vaccine, Busy_vaccine] = calc_metrics(lambd, mu_vaccinate, num_vaccine_booths)
        total_time = W_checkin + W_vaccine + 15
        st.text("Patients can expect to be in the facility for {:0.1f} mins.".format(total_time)) 
        st.text("Vaccinators are busy for approximately {:0.1f} % of time".format(Busy_vaccine*100)) 
        st.text("Checkin personnel are busy for approximately {:0.1f} % of time".format(Busy_checkin*100)) 
        st.text("Approximately {:0.1f} individuals can expect to be waiting in line at check-in".format(Lq_checkin)) 
