import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.append(os.path.abspath(__file__)[:-14])

import PyCORe_main as pcm
import time

start_time = time.time()
Num_of_modes = 512
D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0*75.5e3
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
#Dint[0] = Dint[0]+500e6

dNu_ini = -1e9
dNu_end = 5e9
nn = 2000
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


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




P0 = 0.1### W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = np.sqrt(P0)
#%%
single_ring = pcm.Resonator(PhysicalParameters)

#map2d = single_ring.Propagate_SAM(simulation_parameters, Pump)
#map2d = single_ring.Propagate_SplitStepCLIB(simulation_parameters, Pump,dt=0.5e-3)
map2d = single_ring.Propagate_SAMCLIB(simulation_parameters, Pump,dt=0.5e-3)
#%%
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Pump,dt=1e-3)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%

pcm.Plot_Map(np.fft.ifft(map2d,axis=1),dOm*2/single_ring.kappa)

np.save('map2d_scan',map2d,allow_pickle=True)
np.save('dOm_scan',dOm,allow_pickle=True)
print("--- %s seconds ---" % (time.time() - start_time))