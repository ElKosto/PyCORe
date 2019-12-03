import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('C:/Users/tikan/Documents/Python Scripts/PyCORe')
#sys.path.append('C:/Users/tusnin/Documents/Physics/PhD/epfl/PyCORe')
import PyCORe_main as pcm


Num_of_modes = 64
N_crow = 3

D2 = 4*0.48e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint_1 = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
Dint = [Dint_1,Dint_1,Dint_1]

dNu_ini = -2e8
dNu_end = 5e8
nn = 1000
ramp_stop = 0.99
dOm = 2*np.pi*np.concatenate([np.linspace(dNu_ini,dNu_end, int(nn*ramp_stop)),dNu_end*np.ones(int(np.round((1-ramp_stop)*nn)))])


J_0 = 4.5e9*2*np.pi*np.ones_like(Dint_1)
J = [J_0,J_0]
delta = 0.1e9*2*np.pi
kappa_ex_1 = 25e6*2*np.pi
kappa_ex = [kappa_ex_1, 0., kappa_ex_1]

PhysicalParameters = {'J': J,
                      'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 1000e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 1.35e-6,
                      'kappa_0' : 25e6*2*np.pi,
                      'kappa_ex' : kappa_ex,
                      'Dint' : Dint}

simulation_parameters = {'slow_time' : 1e-5,
                         'detuning_array' : dOm,
                         'noise_level' : 1e-3,
                         'output' : 'map',
                         'absolute_tolerance' : 1e-9,
                         'relative_tolerance' : 1e-9,
                         'max_internal_steps' : 2000}

P0 = 0.004### W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = P0


Seed = Pump/100000

crow = pcm.CROW(PhysicalParameters)

map2d = crow.Propagate_SAM(simulation_parameters, Seed, Pump)
#map2d = single_ring.Propagate_SplitStep(simulation_parameters, Seed, Pump)