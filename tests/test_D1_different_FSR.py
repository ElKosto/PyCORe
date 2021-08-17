#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 09:13:35 2021

@author: tusnin
"""

import matplotlib.pyplot as plt
import numpy as np
import sys, os
curr_dir = os.getcwd()
PyCore_dir = os.path.dirname(curr_dir)
sys.path.append(PyCore_dir)
import PyCORe_main as pcm

import time

start_time = time.time()

Num_of_modes = 2**9
N_crow = 2

FSR = 30e9
D1 = 2*np.pi*FSR
D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2

D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint_single = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint = np.zeros([mu.size,N_crow])
Dint = (Dint_single*np.ones([mu.size,N_crow]).T).T
#Dint[:,1]*=-3

J = 4.5e9*2*np.pi*np.ones([mu.size,(N_crow-1)])
Delta_D1 = np.ones(N_crow)
Delta_D1[0] = 0
Delta_D1[1] = 0.1*D1

dNu_ini = -1e9
dNu_end = 3e9

nn = 20000
ramp_stop = 1.0
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


kappa_ex_ampl = 50e6*2*np.pi
kappa_ex = np.zeros([Num_of_modes,N_crow])
kappa_ex[:,0] = kappa_ex_ampl*np.ones([Num_of_modes])
kappa_ex[:,1] = 0*kappa_ex_ampl*np.ones([Num_of_modes])

Delta = np.zeros([mu.size,(N_crow)])
Delta[:,1] = -2*np.pi*250e9

PhysicalParameters = {'Inter-resonator_coupling': J,
                      'Resonator detunings' : Delta,
                      'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : FSR ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 0.85e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Dint' : Dint,
                      'Delta D1': Delta_D1}

simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-6,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-8,
                         'relative_tolerance' : 1e-8,
                         'max_internal_steps' : 2000}

P0 = .8### W
Pump = np.zeros([len(mu),N_crow],dtype='complex')

Pump[0,0] = np.sqrt(P0)

#%%
crow = pcm.CROW()
crow.Init_From_Dict(PhysicalParameters)
#ev = crow.Linear_analysis()

#%%

map2d = crow.Propagate_SAMCLIB(simulation_parameters, Pump, BC='OPEN')

#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,0])**2,axis=1))
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,1])**2,axis=1))
pcm.Plot_Map(np.fft.ifft(map2d[:,:,0],axis=1),dOm*2/crow.kappa_0)

