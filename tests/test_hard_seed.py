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

map2d_scan = np.zeros([],dtype=complex)#np.load('map2d_scan.npy')
dOm_scan = np.zeros([])
Pump=np.zeros([],dtype=complex)
simulation_parameters={}
single_ring = pcm.Resonator()
#single_ring=pcm.CROW()
simulation_parameters,map2d_scan,dOm_scan,Pump=single_ring.Init_From_File('./data/')

idet = 1100
nn = 10000
dOm = np.ones(nn)*dOm_scan[idet]
simulation_parameters['slow_time']=1e-6
simulation_parameters['detuning_array']=dOm

Seed = map2d[-1,:]
#Seed = map2d_scan[idet,:]#/single_ring.N_points


#%%
#map2d = single_ring.Propagate_SAM(simulation_parameters, Pump)
map2d = single_ring.Propagate_SplitStepCLIB(simulation_parameters, Pump,Seed=Seed,dt=0.5e-3, HardSeed=True)
#map2d = single_ring.Propagate_SAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True,BC='OPEN')
#map2d = single_ring.Propagate_SAMCLIB(simulation_parameters, Pump,Seed=Seed,HardSeed=True)
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Pump,dt=1e-3)
#%%
#plt.figure()
#plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%

#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),dOm*2/single_ring.kappa)
#pcm.Plot_Map(np.fft.ifft(map2d,axis=1),np.arange(nn))
#np.save('map2d_'+str(idet),map2d[:,:],allow_pickle=True)

print("--- %s seconds ---" % (time.time() - start_time))