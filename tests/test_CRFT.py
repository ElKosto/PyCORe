import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.abspath(__file__)[:-19])
#sys.path.append('C:/Users/tusnin/Documents/Physics/PhD/epfl/PyCORe')
import PyCORe_main as pcm

import time

start_time = time.time()

Num_of_modes = 2**7
N_theta = 2**5
N_crow = 200
D2 = 4.1e6*2*np.pi
J = 4.5e6*2*np.pi
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
kappa0=50e6*2*np.pi
kappa_ex=50e6*2*np.pi
kappa= kappa0+kappa_ex
J = 1e9*2*np.pi

zeta_ini = -10
zeta_end = 10

nu_ini = zeta_ini*2/kappa
nu_end = zeta_end*2/kappa

nn = 4000

zeta = np.linspace(zeta_ini,zeta_end,nn)



PhysicalParameters = {'Inter-resonator_coupling': J,
                      'N_res': N_crow,
                      'N_theta': N_theta,
                      'D2': D2,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Number of modes':Num_of_modes}
simulation_parameters = {'slow_time' : 1e-7,
                         'detuning_array' : zeta,
                         'noise_level' : 1e-6,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-8,
                         'relative_tolerance' : 1e-8,
                         'max_internal_steps' : 2000}

f2 = 1.2
f = np.sqrt(f2)

Sin = np.zeros([len(mu),N_theta],dtype='complex')
for ii in range(N_theta):
    Sin[0,ii] = f/np.sqrt(N_theta)

crow = pcm.FieldTheoryCROW(PhysicalParameters)
#%%
map2d = crow.Propagate_SAMCLIB(simulation_parameters, Sin)

#%%
plt.figure()
plt.plot(zeta,np.mean(np.mean(np.abs(map2d[:,:,:])**2,axis=1),axis=1))