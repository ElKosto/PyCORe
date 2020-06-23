import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.append(os.path.abspath(__file__)[:-14])

import PyCORe_main as pcm
import time

start_time = time.time()
Num_of_modes = 512
D2 = 8*0.5e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint[33] = Dint[33]#+500e6

dNu_ini = -3e8
dNu_end = 5e8
nn = 1000
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


PhysicalParameters = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 1000e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 1.35e-6,
                      'kappa_0' : 25e6*2*np.pi,
                      'kappa_ex' : 25e6*2*np.pi,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'electro-optical coupling' : -9*(25e6*2*np.pi)*0,
                         'noise_level' : 1e-7,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-9,
                         'relative_tolerance' : 1e-9,
                         'max_internal_steps' : 2000}




P0 = 0.002### W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = np.sqrt(P0)

single_ring = pcm.Resonator(PhysicalParameters)

#map2d = single_ring.Propagate_SAM(simulation_parameters, Pump)
map2d = single_ring.Propagate_SplitStepCLIB(simulation_parameters, Pump,dt=1e-3)
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Pump,dt=1e-3)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d)**2,axis=1))
#%%

pcm.Plot_Map(np.fft.ifft(map2d,axis=1),dOm*2/single_ring.kappa)

print("--- %s seconds ---" % (time.time() - start_time))