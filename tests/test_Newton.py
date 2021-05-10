#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:55:47 2021

@author: alextusnin
"""

import matplotlib.pyplot as plt
import numpy as np
import sys,os
from scipy.constants import hbar
curr_dir = os.getcwd()
sys.path.append(curr_dir[:-5])

import PyCORe_main as pcm
import time

start_time = time.time()
Num_of_modes = 2**9
D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 75.5e3
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
#Dint[33] = Dint[33]#+500e6

dNu_ini = -1e9
dNu_end = 3e9
nn = 2000
ramp_stop = 0.99
#dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])
dOm = 2*np.pi*np.linspace(dNu_ini,dNu_end,nn)

PhysicalParameters = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 181.7e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 0.85e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : 50e6*2*np.pi,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'electro-optical coupling' : -3*(25e6*2*np.pi)*0,
                         'noise_level' : 1e-8,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-8,
                         'relative_tolerance' : 1e-8,
                         'max_internal_steps' : 2000}




P0 = 0.03### W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = np.sqrt(P0)

single_ring = pcm.Resonator(PhysicalParameters)
A = np.ones(Num_of_modes,dtype=complex)
A = 2*np.exp(-(single_ring.phi-np.pi)**2/0.1)

res,J = single_ring.NewtonRaphson(A,1.4e9,Pump,tol=1e-8,max_iter=100)
plt.figure()
plt.plot(abs(res))
plt.ylim(0,abs(res).max()*1.1)
