import matplotlib.pyplot as plt
import numpy as np
import sys, os
curr_dir = os.getcwd()
PyCore_dir = os.path.dirname(curr_dir)
sys.path.append(PyCore_dir)
import PyCORe_main as pcm

import time

start_time = time.time()

Num_of_modes = 2**9
N_crow = 2

D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2

D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint_single = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint = np.zeros([mu.size,N_crow])
Dint = (Dint_single*np.ones([mu.size,N_crow]).T).T#Making matrix of dispersion with dispersion profile of j-th resonator on the j-th column
#for ll in range(N_crow):
#    Dint[:,ll] = Dint[:,ll]*(-1)**(ll)
#dNu_ini = 8e9
#dNu_end = 10e9



#J = 0.5*50e6*2*np.pi*np.ones([mu.size,(N_crow-1)])
J = 2.0e9*2*np.pi*np.ones([mu.size,(N_crow-1)])
#J = np.zeros(N_crow-1)
#for pp in range(N_crow-1):
#    if pp%2: J[pp] = 4.5e9*2*np.pi
#    else: J[pp] = 0.9e9*2*np.pi

dNu_ini = 0.5e9#-2*J.max()/2/np.pi-10e6
dNu_end = 2*J.max()/2/np.pi#+10e9
#dNu_ini = -1e9
#dNu_end = 1e9
#dNu_ini = -10e9
#dNu_end = -7e9
nn = 2000
ramp_stop = 1.0
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


#delta = 0.1e9*2*np.pi
kappa_ex_ampl = 50e6*2*np.pi
kappa_ex = np.zeros([Num_of_modes,N_crow])
#kappa_ex[:,-1] = 2/5*kappa_ex_ampl*np.ones([Num_of_modes])
kappa_ex[:,0] = kappa_ex_ampl*np.ones([Num_of_modes])
kappa_ex[:,1] = kappa_ex_ampl*np.ones([Num_of_modes])
#for ii in range(N_crow):
#    kappa_ex[:,ii] = kappa_ex_ampl*np.ones([Num_of_modes])
#J = (kappa_ex[:,0]/2-kappa_ex[:,1]/2)/2*2*np.pi*np.ones([mu.size,(N_crow-1)])    
Delta = np.zeros([mu.size,(N_crow)])

#Delta[:,0] = 2*np.pi*1e9*np.ones([Num_of_modes])

PhysicalParameters = {'Inter-resonator_coupling': J,
                      'Resonator detunings' : Delta,
                      'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 181.7e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 0.85e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 1e-6,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-9,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-10,
                         'relative_tolerance' : 1e-12,
                         'max_internal_steps' : 2000}

P0 = 0.3### W
#P0 = 0.006### W
Pump = np.zeros([len(mu),N_crow],dtype='complex')
#for ii in range(N_crow):
#    Pump[0,ii] = np.sqrt(P0/N_crow)#*np.exp(1j*2*np.pi*2*ii/10)
#Pump[0,0] = np.sqrt(P0)
Pump[0,0] = np.sqrt(P0)
#Pump[0,9] = np.sqrt(P0/2)
#Pump = np.concatenate((Pump, 0*Pump))

#%%
crow = pcm.CROW()
crow.Init_From_Dict(PhysicalParameters)
#ev = crow.Linear_analysis()
#%%

#map2d = crow.Propagate_SAMCLIB(simulation_parameters, Pump, BC='OPEN')
map2d = crow.Propagate_PSEUDO_SPECTRAL_SAMCLIB(simulation_parameters, Pump, BC='OPEN', lib='NR')
#map2d = crow.Propagate_SAM(simulation_parameters, Pump)
#%%
plt.figure()
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,0])**2,axis=1))
plt.plot(dOm/2/np.pi,np.mean(np.abs(map2d[:,:,1])**2,axis=1))
pcm.Plot_Map(np.fft.ifft(map2d[:,:,0],axis=1),dOm*2/crow.kappa_0)

#%%
#crow.Save_Data(map2d,Pump,simulation_parameters,dOm,'./data/')
print("--- %s seconds ---" % (time.time() - start_time))
    