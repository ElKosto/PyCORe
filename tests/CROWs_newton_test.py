#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
sys.path.append('/home/alextusnin/Documents/Projects/PyCORe')
import PyCORe_main as pcm
plt.rcParams.update({'font.size': 22})
from datetime import datetime
import time
start_time = time.time()
#%%
def ReadData(src_dir):
    Pump = np.load('Pump.npy')
    map2d_scan = np.load('map2d.npy')
    dOm_scan = np.load('dOm.npy')
    J = np.load('Inter-resonator_coupling.npy')
    Dint = np.load('Dint.npy')
    Dint = np.fft.ifftshift(Dint,axes=0)
    kappa_ex = np.load('kappa_ex.npy')
    PhysPar_src_pd = pd.read_csv("Phys_params.csv",sep='\t')
    PhysPar_src = PhysPar_src_pd.to_dict("split")
    #print(PhysPar_src)
    #PhysPar_src=dict(zip(PhysPar_src_pd['index'],PhysPar_src_pd['data']))
    Delta = np.zeros([Dint[:,0].size,(Dint[0,:].size)]) 
    PhysicalParameters = {'Inter-resonator_coupling': J,
                          'Resonator detunings' : Delta,
                          'n0' : PhysPar_src['data'][0][1],
                          'n2' : PhysPar_src['data'][1][1],
                          'FSR' : PhysPar_src['data'][2][1] ,
                          'w0' : PhysPar_src['data'][3][1],
                          'width' : PhysPar_src['data'][4][1],
                          'height' : PhysPar_src['data'][5][1],
                          'kappa_0' : PhysPar_src['data'][6][1],
                          'kappa_ex' : kappa_ex,
                          'Dint' : Dint}
    return PhysicalParameters, map2d_scan, dOm_scan, J, Dint, Pump,PhysPar_src['data'][9][1]

src_dir = '/home/alextusnin/Documents/Periodic_CROWs/data/24102020_23h25m09s/'
try:
    os.chdir(src_dir)
except:
    sys.exit('No such directory')

[PhysicalParameters, map2d_scan, dOm_scan, J, Dint, Pump,D2] = ReadData(src_dir)
if len(sys.argv)<2:
    i_det = 0
else:
    i_det = int(sys.argv[1])

dOm_fix = dOm_scan[i_det]

#%%

Num_of_modes=Dint[:,0].size
print(PhysicalParameters['kappa_0'])
nn = 20000
ramp_stop = 0.99
dOm = dOm_fix*np.ones(nn)
simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-9,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-8,
                         'relative_tolerance' : 1e-8,
                         'max_internal_steps' : 2000}


Seed = map2d_scan[i_det,:,:]/np.sqrt(Num_of_modes)

#%%
crow = pcm.CROW(PhysicalParameters)
#ev = crow.Linear_analysis()
#%%
#P0 = 0.1### W
#Pump = np.zeros(len(mu),dtype='complex')
#Pump[0] = np.sqrt(P0)

idet=7200
A = np.fft.ifft(map2d_scan[idet,:,:],axis=0)*np.sqrt(Num_of_modes)
dOm = dOm_scan[idet]


#S = Pump/np.sqrt(single_ring.w0*hbar)

#%%
res,rel_diff = crow.NewtonRaphson(A,dOm,Pump,tol=1e-6,max_iter=25)
print("Elapsed time " + str(time.time() - start_time) + "seconds")
#%%
plt.figure()
plt.semilogy(rel_diff,'o',c='k')
#%%%
plt.figure()
plt.plot(abs(res))
#plt.plot(abs(A),c='r')
plt.ylim(0,abs(res).max()*1.1)