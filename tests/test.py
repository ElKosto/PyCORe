### test PyCORe
import PyCORe_main as pcm
import matplotlib.pyplot as plt
import numpy as np

PhysicalParameters = {'n0' : ,
                      'n2' : ,
                      'FSR' : ,
                      'w0' : ,
                      'width' : ,
                      'height' : ,
                      'kappa_0' : ,
                      'kappa_ex' : ,
                      'Dint' : }
Pump = 
Seed = 

single_ring = pcm.Resonator(PhysicalParameters, Seed, Pump)

single_ring.Propagate_SAM()