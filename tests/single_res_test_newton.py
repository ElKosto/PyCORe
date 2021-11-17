#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:55:47 2021

@author: alextusnin
"""

import matplotlib.pyplot as plt
import numpy as np
import sys,os
from scipy.constants import hbar
curr_dir = os.getcwd()
PyCore_dir = os.path.dirname(curr_dir)
sys.path.append(PyCore_dir)

import PyCORe_main as pcm
import time
#%%
start_time = time.time()

map2d_scan=np.load('map2d_scan.npy')
dOm_scan=np.load('dOm_scan.npy')
nn = dOm_scan.size
idet = 731
Num_of_modes = map2d_scan[0,:].size
D2 = 4.1e6#-1*beta2*L/Tr*D1**2 ## From beta2 to D2
D3 = 0*75.5e3
mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
Dint = 2*np.pi*(mu**2*D2/2 + mu**3*D3/6)
#Dint[33] = Dint[33]#+500e6


PhysicalParameters = {'n0' : 1.9,
                      'n2' : 2.4e-19,### m^2/W
                      'FSR' : 181.7e9 ,
                      'w0' : 2*np.pi*192e12,
                      'width' : 1.5e-6,
                      'height' : 0.85e-6,
                      'kappa_0' : 50e6*2*np.pi,
                      'kappa_ex' : 50e6*2*np.pi,
                      'Dint' : Dint}


P0 = 0.1### W
Pump = np.zeros(len(mu),dtype='complex')
Pump[0] = np.sqrt(P0)


#A = np.fft.ifft(map2d_scan[idet,:])#/np.sqrt(Num_of_modes)
A = map2d_scan[idet,:]#/np.sqrt(Num_of_modes)
dOm = dOm_scan[idet]

single_ring = pcm.Resonator()
single_ring.Init_From_Dict(PhysicalParameters)
S = Pump/np.sqrt(single_ring.w0*hbar)

#%%
res,rel_diff = single_ring.NewtonRaphson(A,dOm,Pump,tol=1e-8,max_iter=100)
print("Elapsed time " + str(time.time() - start_time) + "seconds")
#%%
plt.figure()
plt.semilogy(rel_diff,'o',c='k')
#%%%
plt.figure()
plt.plot(abs(res))
#plt.plot(abs(A),c='r')
plt.ylim(0,abs(res).max()*1.1)

#%%
#plt.figure()
#plt.plot(abs(A),c='r')
#plt.ylim(0,abs(A).max()*1.1)