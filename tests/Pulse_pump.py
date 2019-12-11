# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 13:34:18 2019

@author: tikan
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/tikan/Documents/Python Scripts/PyCORe')
#sys.path.append('C:/Users/tusnin/Documents/Physics/PhD/epfl/PyCORe')
import PyCORe_main as pcm

Num_of_modes = 512*2
#Tr = 1./18.2e9#2*np.pi*R*c/n0
#L = 11.9e-3#c/n0*Tr#
###dispersion
#D1 = 2*np.pi*1/Tr
#beta2 = -13e-27
D2 = -3e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 8
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint[133] = Dint[133]+0
#k = 1.75e-5/Tr  #(alpha+theta_ex)/2
#alpha = 1.75e-5
#gamma = 0.000032 # n2*2*np.pi*f_pump/c/Aeff

dNu_ini = -5e8
dNu_end = 45e8
nn = 1000
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


resonator_parameters = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 28e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 1.35e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : 50e6*2*np.pi,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-9,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-9,
                         'relative_tolerance' : 1e-9,
                         'max_internal_steps' : 2000}


P0 = 40### W
tt = np.linspace(-0.5/28e9, 0.5/28e9, Num_of_modes)/1e-12
 
Pump = np.fft.fft(np.sqrt(P0)*np.exp(-(tt+0.5/28e9)**2/5.5**2/2))/Num_of_modes#np.zeros(len(mu),dtype='complex')
#Pump[0] = P0
#Pump = np.fft.fftshift(Pump)
#
#plt.figure()
#plt.plot(np.real(Pump))
#plt.plot(np.imag(Pump))
#plt.plot(np.abs(Pump))
#plt.plot(np.angle(Pump))

single_ring = pcm.Resonator(resonator_parameters)

Seed =  Pump/1000000#single_ring.seed_soliton(Pump, dOm[0])#single_ring.seed_level(Pump, dOm[0])#
#%%
map2d = single_ring.Propagate_SplitStep(simulation_parameters,Pump)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%

pcm.Plot_Map(np.fft.ifft(map2d,axis=1))