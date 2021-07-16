#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:41:40 2020
@author: tusnin
"""


import matplotlib.pyplot as plt
import numpy as np
import sys,os
curr_dir = os.getcwd()
PyCore_dir = os.path.dirname(curr_dir)
sys.path.append(PyCore_dir)

import PyCORe_main as pcm
import time
idet = 700
map2d_scan=np.load('map2d_scan.npy')
map2d=np.load('map2d_'+str(idet)+'.npy')
dOm=np.load('dOm_scan.npy')
nn = map2d[:,0].size

Time=1e-6
hann_window = np.hanning(nn)

N_modes = 512
phi = np.linspace(0,2*np.pi,N_modes)
mu = np.arange(0,N_modes) - N_modes/2
slow_freq = (np.arange(0,nn) - nn/2)/Time
map2d_direct = np.zeros([nn,N_modes],dtype=complex)

for jj in range(nn):
    map2d_direct[jj,:] = np.fft.ifft(map2d[jj,:],axis=0)
for jj in range(0,N_modes):
    map2d_direct[:,jj]*= hann_window
#%%

NDR =np.fft.fftshift(np.fft.fft2(map2d_direct[:,:]))      
max_val =np.max(np.abs(NDR[:,:])) 
fig = plt.figure(figsize=[3.6,2.2],frameon=False,dpi=200)
ax = fig.add_subplot(1,1,1)
colbar=ax.pcolormesh(mu,-slow_freq/1e9,10*np.log10(np.abs(NDR)**2/max_val**2),cmap='afmhot',rasterized=True)
colbar.set_clim(-150,0)
ax.set_xlim(-200,200)
#ax.set_xticks([-256,0,255])

#ax.set_xticklabels([r'$-256$','0',r'$256$'])
ax.set_xlabel('Relative mode number 'r'$\mu$')

#ax.set_ylim(-7500,7500)
#ax.set_yticks([-300,0,300])


colorbar=fig.colorbar(colbar,ax=ax)
colorbar.set_label("PSD (dB)")
ax.set_ylabel(r'$D_{int}(\mu)$ (GHz)')
plt.tight_layout()
plt.savefig("NDR_"+str(idet)+'.pdf')
plt.show()

#%%

field =np.fft.ifft(map2d[:,:],axis = 1)
max_val =np.max(np.abs(field[:,:])) 
fig = plt.figure(figsize=[3.6,2.2],frameon=False,dpi=200)
ax = fig.add_subplot(1,1,1)
colbar=ax.pcolormesh(phi-np.pi,Time/10*np.linspace(0,1,2000)/1e-6,(np.abs(field[2000:4000,:])**2/max_val**2),rasterized=True)
colbar.set_clim(0,1)
ax.set_xlim(-np.pi,np.pi)
ax.set_xticks([-np.pi,0,np.pi])
ax.set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])

#ax.set_xticklabels([r'$-256$','0',r'$256$'])
ax.set_xlabel(r'$\varphi$')
ax.set_ylabel(r'Time ($\mu$s)')

#ax.set_ylim(-7500,7500)
#ax.set_yticks([-300,0,300])


colorbar=fig.colorbar(colbar,ax=ax)
colorbar.set_label("Intensity (arb. units)")

plt.tight_layout()
plt.savefig("field_"+str(idet)+'.pdf')
plt.show()

#%%

spectrum = np.fft.fftshift(map2d[:,:],axes=1)
max_val =np.max(np.abs(spectrum[:,:])) 
fig = plt.figure(figsize=[3.6,2.2],frameon=False,dpi=200)
ax = fig.add_subplot(1,1,1)
colbar=ax.pcolormesh(mu,Time/10*np.linspace(0,1,2000)/1e-6,10*np.log10(np.abs(spectrum[2000:4000,:])**2/max_val**2),rasterized=True)
colbar.set_clim(-150,0)
ax.set_xlim(-200,200)
#ax.set_xticks([-np.pi,0,np.pi])
#ax.set_xticklabels([r'$-\pi$',r'$0$',r'$\pi$'])

#ax.set_xticklabels([r'$-256$','0',r'$256$'])
ax.set_xlabel(r'$\mu$')
ax.set_ylabel(r'Time ($\mu$s)')

#ax.set_ylim(-7500,7500)
#ax.set_yticks([-300,0,300])


colorbar=fig.colorbar(colbar,ax=ax)
colorbar.set_label("Spectrum (dB)")

plt.tight_layout()
plt.savefig("spectrum"+str(idet)+'.pdf')
plt.show()