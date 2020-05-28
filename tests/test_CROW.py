import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.abspath(__file__)[:-19])
#sys.path.append('C:/Users/tusnin/Documents/Physics/PhD/epfl/PyCORe')
import PyCORe_main as pcm


Num_of_modes = 2**5
N_crow = 2

D2 = 3e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint_single = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint = np.zeros([mu.size,N_crow])
Dint = (Dint_single*np.ones([mu.size,N_crow]).T).T#Making matrix of dispersion with dispersion profile of j-th resonator on the j-th column

dNu_ini = 4e9
dNu_end = 5e9
nn = 100
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


J = 4.5e9*2*np.pi*np.ones(mu.size*(N_crow-1))

#delta = 0.1e9*2*np.pi
kappa_ex_ampl = 50e6*2*np.pi
kappa_ex = kappa_ex_ampl*np.ones([Num_of_modes,N_crow])


PhysicalParameters = {'Inter-resonator_coupling': J,
                      'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 180e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 0.85e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 5e-7,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-8,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-9,
                         'relative_tolerance' : 1e-9,
                         'max_internal_steps' : 2000}

P0 = 0.3### W
Pump = np.zeros([len(mu),N_crow],dtype='complex')
Pump[0,0] = np.sqrt(P0)
#Pump = np.concatenate((Pump, 0*Pump))


crow = pcm.CROW(PhysicalParameters)
#%%
map2d = crow.Propagate_SplitStep(simulation_parameters, Pump)
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Seed, Pump)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,0])**2,axis=1))
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,1])**2,axis=1))



    