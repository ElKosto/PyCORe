#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:41:40 2020

@author: tusnin
"""


import matplotlib.pyplot as plt
import numpy as np
import sys,os
sys.path.append(os.path.abspath(__file__)[:-13])

import PyCORe_main as pcm
import time

map2d_scan=np.load('./data/map2d.npy')
map2d=np.load('map2d_12500.npy')
dOm=np.load('./data/dOm.npy')
nn = map2d[:,0,0].size

Time=1e-6
hann_window = np.hanning(nn)

N_modes = map2d[0,:,0].size
phi = np.linspace(0,2*np.pi,N_modes)
mu = np.arange(0,N_modes) - N_modes/2
slow_freq = (np.arange(0,nn) - nn/2)/Time
map2d_direct = np.zeros([nn,N_modes],dtype=complex)

for jj in range(nn):
    map2d_direct[jj,:] = np.fft.ifft(map2d[jj,:,0],axis=0)
for jj in range(0,N_modes):
    map2d_direct[:,jj]*= hann_window

#%%

NDR =np.fft.fftshift(np.fft.fft2(map2d_direct[:,:]))      
max_val =np.max(abs(NDR)**2) 
fig = plt.figure(figsize=[3.6,2.2],frameon=True,dpi=300)
ax = fig.add_subplot(1,1,1)
colbar=ax.pcolormesh(mu,-slow_freq/1e9,10*np.log10(np.abs(NDR)**2/max_val),cmap='afmhot',rasterized=True)
colbar.set_clim(-250,0)
#ax.set_xlim(-150,150)
#ax.set_xticks([-256,0,255])

#ax.set_xticklabels([r'$-256$','0',r'$256$'])
ax.set_xlabel('Relative mode number 'r'$\mu$')

#ax.set_ylim(-7500,7500)
#ax.set_yticks([-300,0,300])


colorbar=fig.colorbar(colbar,ax=ax)
colorbar.set_label("PSD (dB)")
ax.set_ylabel('Slow frequency (arb. units)')
plt.show()