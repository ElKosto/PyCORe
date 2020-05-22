import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/tikan/Documents/Python Scripts/PyCORe')
#sys.path.append('C:/Users/tusnin/Documents/Physics/PhD/epfl/PyCORe')
import PyCORe_main as pcm


Num_of_modes = 128
N_crow = 2

D2 = 3e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint_1 = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint = [Dint_1]*N_crow

dNu_ini = -2e9
dNu_end = 5e9
nn = 1000
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


J_0 = 4.5e9*2*np.pi*np.ones_like(Dint_1)
J = [J_0]*(N_crow-1)
#delta = 0.1e9*2*np.pi
kappa_ex_1 = 100e6*2*np.pi*np.ones(Num_of_modes)
kappa_ex = [kappa_ex_1, kappa_ex_1/5]

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

simulation_parameters = {'slow_time' : 1e-8,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-5,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-9,
                         'relative_tolerance' : 1e-9,
                         'max_internal_steps' : 2000}

P0 = .5### W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = np.sqrt(P0)
Pump = np.concatenate((Pump, 0*Pump))

Seed = Pump/10000

crow = pcm.CROW(PhysicalParameters)

map2d = crow.SAM_CROW(simulation_parameters, Pump, Seed)
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Seed, Pump)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:len(mu)])**2,axis=1))
plt.plot(dOm//2/np.pi,np.mean(np.abs(map2d[:,len(mu):])**2,axis=1))
#%% 
ind = 400
plt.figure()### spectra
plt.subplot(211)
plt.plot(np.linspace(-np.pi,np.pi,len(mu)),np.abs(np.fft.ifft(map2d[ind,:len(mu)]))**2)
plt.plot(np.linspace(-np.pi,np.pi,len(mu)),np.abs(np.fft.ifft(map2d[ind,len(mu):]))**2)
plt.subplot(212)
m_norm = max(np.abs(map2d[ind,:len(mu)])**2)
plt.plot(10*np.log10(np.abs(map2d[ind,:len(mu)])**2/m_norm))
plt.plot(10*np.log10(np.abs(map2d[ind,len(mu):])**2/m_norm))
plt.ylim(-100,0)



    