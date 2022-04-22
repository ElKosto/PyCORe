#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:37:11 2022

@author: alextusnin
based on parameters taken from DOI: 10.1126/science.abh2076
"""

import matplotlib.pyplot as plt
import numpy as np
import sys,os
curr_dir = os.getcwd()
PyCore_dir = os.path.dirname(curr_dir)
sys.path.append(PyCore_dir)

import PyCORe_main as pcm
import time



start_time = time.time()
Num_of_modes = 2**9
D2 = 1*4e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
#D3 = -25e3
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2)
Dint[0] = Dint[0]-5* 150e6*2*np.pi


dNu_ini = -1.0e9+5* 150e6
dNu_end = 1.5e9+5* 150e6

#dNu_ini = -1e9
#dNu_end = 5e9
#dNu_ini = -.5e9
#dNu_end = .5e9

nn = 4000
ramp_stop = 1
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])
#%%

PhysicalParameters = {'n0' : 2.0,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 100.0e9 ,
                      'w0' : 2*np.pi*193e12,
                      'width' : 1.*1e-6,
                      'height' : 0.997*1e-6,
                      'kappa_0' : 150e6*2*np.pi,
                      'kappa_ex' : 0*50e6*2*np.pi,
                      'kappa_sc' : 20*1e6*2*np.pi,
                      'Dint' : Dint}
LaserParameters = {'alpha_h': 5,#Linewidth enhancement factor
                   'a': 2*np.pi*1*1e4, #Differential gain
                   'N0': 1e24, #Carrier density at transparency
                   'kappa_laser': 2*np.pi*1e11, #Laser cavity loss rate
                   'kappa_inj': 4.75*1e8*2*np.pi, #Laser-microresonator coupling rate
                   'I': 0.24, #Laser biased current
                   'gamma': 2*np.pi*1e9, #Carrier recombination rate
                   'V': 2*1e-16, #Volume of active section
                   'eta': 2.676 * 1e-7,# Conversion factor
                   'theta' :0.95*np.pi,# rad, optical feedback
                   'zeta' : 5*1e11 #Hz/A, Current-frequency tuning coefficient
    }
simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-12,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-8,
                         'relative_tolerance' : 1e-8,
                         'max_internal_steps' : 2000}





single_ring = pcm.SiL_Resonator()
single_ring.Init_From_Dict(PhysicalParameters,LaserParameters)
#%%
map2d, CCW_res, Laser_field, N = single_ring.Propagate_PseudoSpectralSAMCLIB(simulation_parameters,dt=1e-3)
#%%
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Pump,dt=1e-3)
#%%
#plt.figure()
#plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%
plt.rcParams.update({'font.size': 10})
Nm = map2d[0,:].size
fig = plt.figure(figsize=[3.6,3.6],frameon=False,dpi=200)
ax = fig.add_subplot(1,1,1)
ax1=ax.twinx()
maxval = np.mean(np.abs(map2d[:,:]/np.sqrt(Nm))**2,axis=1).max()
ax.plot(dOm/2/np.pi/1e9,np.mean(np.abs(map2d[:,:]/np.sqrt(Nm))**2,axis=1)/maxval,label='generated power',c='k')
ax1.plot(dOm/2/np.pi/1e9,np.unwrap(np.angle(CCW_res)),label='phase in CCW',c='g')
ax1.plot(dOm/2/np.pi/1e9,np.unwrap(np.angle(map2d[:,0])),c='r',label='phase 0-th comb lane')
ax1.plot(dOm/2/np.pi/1e9,np.unwrap(np.angle(Laser_field)),label='laser phase')
ax.set_xlim(dOm.min()/2/np.pi/1e9,dOm.max()/2/np.pi/1e9)
ax.set_xlabel('Laser detuning (GHz)')
ax.set_ylabel('Generated light (arb. units)')
ax.legend()
ax1.legend()
ax.grid(True)
plt.tight_layout()
#plt.savefig('trans_trace_pres.png')
plt.show()

#%%

pcm.Plot_Map(np.fft.ifft(map2d,axis=1),np.arange(nn))

#np.save('map2d_scan',map2d,allow_pickle=True)
#np.save('dOm_scan',dOm,allow_pickle=True)

#%%
#single_ring.Save_Data(map2d,Pump,simulation_parameters,dOm,'./data/')
print("--- %s seconds ---" % (time.time() - start_time))