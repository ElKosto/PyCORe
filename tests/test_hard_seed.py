#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.append(os.path.abspath(__file__)[:-23])

#%%
import PyCORe_main as pcm
import time

start_time = time.time()
Num_of_modes = 512
D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 75.5e3
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint[33] = Dint[33]#+500e6

dNu_ini = -1e9
dNu_end = 3e9
nn = 2000
ramp_stop = 0.99
#dOm_scan = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])

map2d_scan = np.zeros([],dtype=complex)#np.load('map2d_scan.npy')
dOm_scan = np.zeros([])
Pump=np.zeros([],dtype=complex)
simulation_parameters={}
single_ring = pcm.Resonator()
#single_ring=pcm.CROW()
simulation_parameters,map2d_scan,dOm_scan,Pump=single_ring.Init_From_File('./data/')

idet = 1000
nn = 10000
dOm = np.ones(nn)*dOm_scan[idet]
simulation_parameters['slow_time']=1e-6
simulation_parameters['detuning_array']=dOm


Seed = map2d_scan[idet,:]#/single_ring.N_points


#%%
#map2d = single_ring.Propagate_SAM(simulation_parameters, Pump)
#map2d = single_ring.Propagate_SplitStepCLIB(simulation_parameters, Pump,Seed=Seed,dt=1e-3, HardSeed=True)
map2d = single_ring.Propagate_SAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True)
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Pump,dt=1e-3)
#%%
#plt.figure()
#plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%

#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),dOm*2/single_ring.kappa)
#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),np.arange(nn))
#np.save('map2d_'+str(idet),map2d[:,:],allow_pickle=True)

print("--- %s seconds ---" % (time.time() - start_time))