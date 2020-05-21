import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.append(os.path.abspath(__file__)[:-23])
print(os.path.abspath(__file__)[:-23])
import PyCORe_main as pcm
from scipy.constants import c, hbar

Num_of_modes = 512
D2 = 4e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint[33] = Dint[33]#+500e6

dNu_ini = -2e8
dNu_end = 5e8
nn = 4000
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])

### 1 THz SiN
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
                         'electro-optical coupling' : 0.,
                         'noise_level' : 1e-6,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-5,
                         'relative_tolerance' : 1e-5,
                         'max_internal_steps' : 2000}

#%%
P0 = 0.003 #W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = np.sqrt(P0)

single_ring = pcm.Resonator(PhysicalParameters)


#%%
# map2d_sam = single_ring.Propagate_SAM(simulation_parameters, Pump)
# map2d_sstep = single_ring.Propagate_SplitStep(simulation_parameters, Pump, dt=5e-4)
map2d_sstepC = single_ring.Propagate_SplitStepCLIB(simulation_parameters, Pump, dt=5e-4)
#%%
plt.figure()
plt.plot(dOm*2/single_ring.kappa,np.mean(np.abs(map2d_sam)**2,axis=1))
plt.plot(dOm*2/single_ring.kappa,np.mean(np.abs(map2d_sstep)**2,axis=1))
plt.plot(dOm*2/single_ring.kappa,np.mean(np.abs(map2d_sstepC)**2,axis=1))
# f_sq = P0*(np.sqrt(1./(hbar*single_ring.w0))*np.sqrt(8*single_ring.g0*single_ring.kappa_ex/single_ring.kappa**3))**2
# sol_end = f_sq*np.pi**2/8
# plt.plot([sol_end,sol_end], [0, np.max(np.mean(np.abs(map2d_sstep)**2))])
# map2d_sstep
# MI_beg = -np.sqrt(f_sq-1)+1
# MI_end = np.sqrt(f_sq-1)+1
# plt.plot([MI_beg,MI_beg], [0, np.max(np.mean(np.abs(map2d_sstep)**2))],'k')
# plt.plot([MI_end,MI_end], [0, np.max(np.mean(np.abs(map2d_sstep)**2))],'k')
#%%
# pcm.Plot_Map(np.fft.ifft(map2d_sstepC,axis=1), 2*dOm/single_ring.kappa)
# pcm.Plot_Map(np.fft.ifft(map2d_sstep,axis=1), 2*dOm/single_ring.kappa)

#%%
# dOm_hs = np.ones_like(dOm)*dOm[2500]
# Seed_func =  single_ring.seed_soliton(Pump, dOm_hs[0])#single_ring.seed_level(Pump, dOm[0])#
# simulation_parameters_hs = {'slow_time' : 1e-6,
#                          'detuning_array' : dOm_hs,
#                          'electro-optical coupling' : 0.,
#                          'noise_level' : 1e-6,
#                          'output' : 'map',
#                          'absolute_tolerance' : 1e-5,
#                          'relative_tolerance' : 1e-5,
#                          'max_internal_steps' : 2000}


# map_hs = single_ring.Propagate_SplitStep(simulation_parameters_hs, Pump, Seed=Seed_func, dt=5e-4)
#%%
# plt.figure()
# plt.plot(dOm*2/single_ring.kappa,np.mean(np.abs(map2d_sstep)**2,axis=1))
# plt.plot(dOm*2/single_ring.kappa,np.mean(np.abs(map_hs)**2,axis=1))
#%%
# pcm.Plot_Map(np.fft.ifft(map_hs,axis=1), 2*dOm/single_ring.kappa)

