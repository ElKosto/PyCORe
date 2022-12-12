import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode,solve_ivp, ode
from scipy.sparse.linalg import expm
from scipy.sparse.linalg import inv as inv_sparse
from scipy.sparse.linalg import spsolve as solve_sparse
from scipy.linalg import dft
from scipy.linalg import solve as solve_dense
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.constants import pi, c, hbar, e
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
import time
import sys, os
from scipy.sparse import block_diag,identity,diags, eye, csc_matrix, dia_matrix, isspmatrix
from scipy.sparse.linalg import eigs as scp_eigs
import ctypes
from scipy.linalg import eig, inv, solve, lu_factor, lu_solve
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm

class Resonator:
    
    def __init__(self):
        self.n0 = 0
        self.n2 = 0
        self.FSR = 0
        self.w0 = 0
        self.width = 0
        self.height = 0
        self.kappa_0 = 0
        self.kappa_ex = 0
        self.Dint = np.array([])
        #Auxiliary physical parameters
        self.Tr = 0
        self.Aeff = 0
        self.Leff = 0
        self.Veff = 0
        self.g0 = 0
        self.gamma = 0
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
        self.tau_r = 0
    
        self.phi = np.array([])
        
        self.D2 = 0
        self.D3 = 0
        
        self.D2_mod = 0
        
        self.n2t = 0
        self.t_th=0
        
        self.J_EO = 0
        self.D=np.zeros([0],dtype=complex)
        self.FirstDmat=np.zeros([0],dtype=complex)
    
    def Init_From_File(self,data_dir):
        simulation_parameters={}
        map2d=np.array([],dtype=complex)
        Pump=np.array([],dtype=complex)
        dOm=np.array([])
       
        for file in os.listdir(data_dir+'class_parameters/'):
            if file.endswith('.npy'):
                key = os.path.splitext(file)[0]
                print(file + " is open")
                self.__dict__[key] = np.load(data_dir+'class_parameters/'+file)
        for file in os.listdir(data_dir+'sim_parameters/'):
            if file.endswith('.npy'):
                key = os.path.splitext(file)[0]
                simulation_parameters[key] = np.load(data_dir+'sim_parameters/'+file)
        map2d=np.load(data_dir+'map2d.npy')
        dOm=np.load(data_dir+'dOm.npy')
        Pump=np.load(data_dir+'Pump.npy')
        
        return simulation_parameters, map2d, dOm, Pump
    def Init_From_Dict(self, resonator_parameters):
        #Physical parameters initialization
        self.n0 = resonator_parameters['n0']
        self.n2 = resonator_parameters['n2']
        self.FSR = resonator_parameters['FSR']
        self.w0 = resonator_parameters['w0']
        self.width = resonator_parameters['width']
        self.height = resonator_parameters['height']
        self.kappa_0 = resonator_parameters['kappa_0']
        self.kappa_ex = resonator_parameters['kappa_ex']
        self.Dint = np.fft.ifftshift(resonator_parameters['Dint'])
        self.tau_r = resonator_parameters['Raman time']
        
        #Auxiliary physical parameters
        self.Tr = 1/self.FSR #round trip time
        self.Aeff = self.width*self.height 
        self.Leff = c/self.n0*self.Tr 
        self.Veff = self.Aeff*self.Leff 
        self.g0 = hbar*self.w0**2*c*self.n2/self.n0**2/self.Veff
        self.gamma = self.n2*self.w0/c/self.Aeff
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
        self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
        self.phi = np.linspace(0,2*np.pi,self.N_points)
        self.D = self.DispersionMatrix(order=0)
        def func(x, a, b, c, d):
            return a + x*b + c*x**2/2 + d*x**3/6
        popt, pcov = curve_fit(func, self.mu, self.Dint)
        self.D2 = popt[2]
        self.D3 = popt[3]
        
        if 'Modulated D2' in resonator_parameters.keys():
            self.D2_mod = resonator_parameters['Modulated D2']
        else:
            self.D2_mod = 0
        
        if 'T thermal' in resonator_parameters.keys():
            self.t_th = resonator_parameters['T thermal']
            self.n2t = resonator_parameters['n2 thermal']
        
        if 'electro-optical coupling' in resonator_parameters.keys():
            self.J_EO =  resonator_parameters['electro-optical coupling']
        else:
            self.J_EO = 0
        
    def Save_Data(self,map2d,Pump,Simulation_Params,dOm=[0],directory='./'):
        params = self.__dict__
        try: 
            os.mkdir(directory+'class_parameters/')
            os.mkdir(directory+'sim_parameters/')
        except:
            pass
        for key in params.keys():
            np.save(directory+'class_parameters/'+key+'.npy',params[key])
        for key in Simulation_Params:
            np.save(directory+'sim_parameters/'+key+'.npy',Simulation_Params[key])
        np.save(directory+'map2d.npy',map2d)
        np.save(directory+'dOm.npy',dOm)
        np.save(directory+'Pump.npy',Pump)
        
        #print(params.keys())
        
    
    def noise(self, a):
#        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)
        return a*(np.random.uniform(-1,1,self.N_points) + 1j*np.random.uniform(-1,1,self.N_points))

    #   Propagate Using the Step Adaptive  Method
    def Propagate_SAM(self, simulation_parameters, Pump, Seed=[0], Normalized_Units=False):
        start_time = time.time()

        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa)
            ### renormalization
            T_rn = (self.kappa/2)*T
            f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            J = self.J_EO
            J*=2/self.kappa
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        else:
            pump = Pump
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)
            else:
                seed = Seed
            T_rn = T
            f0 = pump
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]) + ',' +str(detuning[-1])+ ']')
            detuning*=self.kappa/2
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        ### define the rhs function
        def LLE_1d(Time, A):
            A = A - noise_const#self.noise(eps)
            A_dir = np.fft.ifft(A)*len(A)## in the direct space
            dAdT =  -1*(1 + 1j*(self.Dint + dOm_curr)*2/self.kappa)*A + 1j*np.fft.fft(A_dir*np.abs(A_dir)**2)/len(A) + 1j*np.fft.fft(J*2/self.kappa*np.cos(self.phi)*A_dir/self.N_points) + f0#*len(A)
            return dAdT
        
        t_st = float(T_rn)/len(detuning)
        r = complex_ode(LLE_1d).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        #r = ode(LLE_1d).set_integrator('zvode', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        r.set_initial_value(seed, 0)# seed the cavity
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = seed
        #printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50, fill='elapsed time = ' + str((time.time() - start_time)) + ' s')
        for it in range(1,len(detuning)):
            self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
            #self.print('elapsed time = ', (time.time() - start_time))
            dOm_curr = detuning[it] # detuning value
            sol[it] = r.integrate(r.t+t_st)
            
        if out_param == 'map':
            if Normalized_Units == False :
                return sol/np.sqrt(2*self.g0/self.kappa)
            else:
                detuning/=self.kappa/2
                return sol
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
    def Propagate_SAM_NEW(self, simulation_parameters, Pump, Seed=[0], Normalized_Units=False):
        start_time = time.time()

        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        if 'electro-optical coupling' in simulation_parameters.keys():
            J =  simulation_parameters['electro-optical coupling']
        else:
            J = 0
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa)
            ### renormalization
            T_rn = (self.kappa/2)*T
            f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            J*=2/self.kappa
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        else:
            pump = Pump
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)
            else:
                seed = Seed
            T_rn = T
            f0 = pump
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]) + ',' +str(detuning[-1])+ ']')
            detuning*=self.kappa/2
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        ### define the rhs function
        # def LLE_1d(Time, A):
        #     #A = A - noise_const#self.noise(eps)
        #     A_dir = np.fft.ifft(A)*len(A)## in the direct space
        #     dAdT =  -1*(1 + 1j*(self.Dint + dOm_curr)*2/self.kappa)*A + 1j*np.fft.fft(A_dir*np.abs(A_dir)**2)/len(A)  + f0#*len(A)
        #     return dAdT
        disp_operator = self.Dint*2/self.kappa
        f0_dir = np.fft.ifft(f0)*self.N_points
        print('f0^2 = ' + str(np.round(max(abs(f0_dir)**2), 2)))
        def LLE_1d(t,A):
            #A+=noise_const
            dAdt = np.fft.ifft((-1j*disp_operator-(self.kappa + 1j*dOm_curr*2)/self.kappa)*(np.fft.fft(A)) ) +1j*np.abs(A)**2*A+f0_dir
            return dAdt
        
        t_st = float(T_rn)/len(detuning)
        #r = complex_ode(LLE_1d).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        #r = ode(LLE_1d).set_integrator('zvode', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        #r.set_initial_value(seed, 0)# seed the cavity
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = np.fft.ifft(seed+noise_const)#/self.N_points
        print(np.sum(abs(sol[0,:]))**2)
        #printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50, fill='elapsed time = ' + str((time.time() - start_time)) + ' s')
        T_span = np.linspace(0,T_rn,len(detuning))
        for it in range(1,len(detuning)):
            self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
            #self.print('elapsed time = ', (time.time() - start_time))
            dOm_curr = detuning[it] # detuning value
            #sol[it] = r.integrate(r.t+t_st)
            sol[it,:]=(solve_ivp( LLE_1d, t_span=[T_span[it-1],T_span[it]], y0=sol[it-1,:], method='DOP853', t_eval=[T_span[it]], atol=abtol, rtol=reltol,max_step=nmax,first_step=1e-3, min_step=1e-5)).y.T
            #print(np.sum(abs(sol[it,:]))**2)
            
        if out_param == 'map':
            if Normalized_Units == False :
                return np.fft.fft(sol,axis=1)/np.sqrt(2*self.g0/self.kappa)/self.N_points
            else:
                detuning/=self.kappa/2
                return np.fft.fft(sol,axis=1)
        elif out_param == 'fin_res':
            return np.fft.fft(sol[-1, :])/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')   
    def Propagate_SplitStep(self, simulation_parameters, Pump, Seed=[0], dt=5e-4, Normalized_Units=False):
        start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)*np.sqrt(2*self.g0/self.kappa)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa)
            ### renormalization
            T_rn = (self.kappa/2)*T
            f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            J*=2/self.kappa
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        else:
            pump = Pump
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0],Normalized_Units)
            else:
                seed = Seed
            T_rn = T
            f0 = pump
            print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]) + ',' +str(detuning[-1])+ ']')
            detuning*=self.kappa/2
        
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        
        print('J = ' + str(J))
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = seed
        #f0 = np.fft.ifft(f0)*self.N_points
        #f0*=self.N_points
        self.printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for it in range(1,len(detuning)):
            noise_const = self.noise(eps)
            self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
            dOm_curr = detuning[it] # detuning value
            t=0
            buf = sol[it-1,:]
            buf-=noise_const
            buf = np.fft.ifft(buf)*len(buf)
            while t<t_st:
                
                # First step
                
                #buf = np.fft.fft(np.exp(dt*(1j*np.abs(buf)**2+1j*J*(np.cos(self.phi) + 0.*np.sin(2*self.phi)) + f0/buf))*buf)
                buf = np.fft.fft(np.exp(dt*(1j*np.abs(buf)**2+1j*J*(np.cos(self.phi) + 0.*np.sin(2*self.phi))))*buf)
                #second step
                
                #buf = np.fft.ifft(np.exp(-dt *(1+1j*(self.Dint + dOm_curr)*2/self.kappa )) *buf)
                buf = np.fft.ifft(np.exp(-dt *(1+1j*(self.Dint + dOm_curr)*2/self.kappa )) *buf + f0*self.N_points/(-1-1j*(self.Dint + dOm_curr)*2/self.kappa)*(np.exp(dt*(-1-1j*(self.Dint + dOm_curr)*2/self.kappa)) -1.))
                
                t+=dt
            sol[it,:] = np.fft.fft(buf)/len(buf)
            #sol[it,:] = buf
            
        if out_param == 'map':
            if Normalized_Units == False :
                return sol/np.sqrt(2*self.g0/self.kappa)
            else:
                detuning/=self.kappa/2
                return sol
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter') 
            
    def Propagate_SplitStepCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4, HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)/self.N_points
        ### renormalization
        T_rn = (self.kappa/2)*T
        f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
        j = J/self.kappa*2
        print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
        print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        print('J = ' + str(j))
        #noise_const = self.noise(eps) # set the noise level
        #nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)
        
        #%% crtypes defyning
        if self.D2_mod == 0:
            LLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_lle_core.so')
        else:
            double_p=ctypes.POINTER(ctypes.c_double)
            LLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_lle_core_faraday.so')
            In_D2_mod = np.array(2/self.kappa*self.D2_mod*self.mu**2,dtype=ctypes.c_double)
            In_D2_mod_p = In_D2_mod.ctypes.data_as(double_p)
            
            In_FSR = ctypes.c_double(self.FSR)
            In_kappa = ctypes.c_double(self.kappa)
        LLE_core.PropagateSS.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)*(len(seed))
        #A = seed
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_f_RE = np.array(np.real(f0),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0),dtype=ctypes.c_double)
        In_J = ctypes.c_double(j)
        
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        In_res_RE = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.D2_mod ==0:
            LLE_core.PropagateSS(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            LLE_core.PropagateSS(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_D2_mod_p, In_FSR, In_kappa, In_Ndet, In_Nt, In_dt, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = (In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])*self.N_points
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
        
    def Propagate_SAMCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4,HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
        ### renormalization
        T_rn = (self.kappa/2)*T
        f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3))*self.N_points
        j = J/self.kappa*2
        print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
        print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        print('J = ' + str(j))
        #noise_const = self.noise(eps) # set the noise level
        #nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)/self.N_points
        
        #%% crtypes defyning
        LLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_lle_core.so')
        if self.n2t==0:
            LLE_core.PropagateSAM.restype = ctypes.c_void_p
        else:
            LLE_core.PropagateThermalSAM.restype = ctypes.c_void_p
        
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)#*self.N_points
        
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_f_RE = np.array(np.real(f0),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0),dtype=ctypes.c_double)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_J = ctypes.c_double(j)
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.n2t==0:
            LLE_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            LLE_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_t_th, In_kappa, In_n2, In_n2t, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
            
   
    def Propagate_PseudoSpectralSAMCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4,HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  self.J_EO
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
        ### renormalization
        T_rn = (self.kappa/2)*T
        f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3))*self.N_points
        j = J/self.kappa*2
        print('f0^2 = ' + str(np.round(max(abs(f0)**2), 2)))
        print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        print('J = ' + str(j))
        #noise_const = self.noise(eps) # set the noise level
        #nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol[0,:] = (seed)/self.N_points
        
        #%% crtypes defyning
        LLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_lle_core.so')
        LLE_core.Propagate_PseudoSpectralSAM.restype = ctypes.c_void_p
        if self.tau_r !=0:
            LLE_core.Propagate_PseudoSpectralSAM_Raman.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)#*self.N_points
        
        #plt.plot(abs(A))
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_tau_r_mu = np.array(self.mu*self.tau_r*self.FSR*2*np.pi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_f_RE = np.array(np.real(f0),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0),dtype=ctypes.c_double)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_J = ctypes.c_double(j)
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            
        if self.n2t!=0:
            In_kappa = ctypes.c_double(self.kappa_0+self.kappa_ex)
            In_t_th = ctypes.c_double(self.t_th)
            In_n2 = ctypes.c_double(self.n2)
            In_n2t = ctypes.c_double(self.n2t)
            
            
            
        In_res_RE = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_tau_r_mu_p = In_tau_r_mu.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        if self.n2t==0:
            if self.tau_r == 0:
                LLE_core.Propagate_PseudoSpectralSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            else:
                LLE_core.Propagate_PseudoSpectralSAM_Raman(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_tau_r_mu_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        else:
            #LLE_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_t_th, In_kappa, In_n2, In_n2t, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
            pass
        
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = np.fft.fft(In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
            
#%%            
              
 
    #%%

    def seed_level (self, pump, detuning, Normalized_Units=False):
        if Normalized_Units == False:
            f_norm = pump*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
            detuning_norm  = detuning*2/self.kappa
            stat_roots = np.roots([1, -2*detuning_norm, (detuning_norm**2+1), -abs(f_norm[0])**2])
            ind_roots = [np.imag(ii)==0 for ii in stat_roots]
            res_seed = np.zeros_like(f_norm)
            res_seed[0] = abs(np.min(stat_roots[ind_roots]))**.5/np.sqrt(2*self.g0/self.kappa)
        else:
            f_norm = pump
            detuning_norm  = detuning
            stat_roots = np.roots([1, -2*detuning_norm, (detuning_norm**2+1), -abs(f_norm[0])**2])
            ind_roots = [np.imag(ii)==0 for ii in stat_roots]
            res_seed = np.zeros_like(f_norm)
            res_seed[0] = abs(np.min(stat_roots[ind_roots]))**.5
        return res_seed
    
    def seed_soliton(self, pump, detuning):
        fast_t = np.linspace(-pi,pi,len(pump))*np.sqrt(self.kappa/2/self.D2)
        f_norm = abs(pump[0]*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3))
        detuning_norm  = detuning*2/self.kappa
        stat_roots = np.roots([1, -2*detuning_norm, (detuning_norm**2+1), -abs(f_norm)**2])
        
        ind_roots = [np.imag(ii)==0 for ii in stat_roots]
        B = np.sqrt(2*detuning_norm)
        print(detuning_norm)
        print(f_norm)
        level = np.min(np.abs(stat_roots[ind_roots]))
        print(level)
        return np.fft.fft(level**.5*np.exp(1j*np.arctan((detuning_norm-level)/f_norm)) + B*np.exp(1j*np.arccos(2*B/np.pi/f_norm))*np.cosh(B*fast_t)**-1)/np.sqrt(2*self.g0/self.kappa)/len(pump)
        
        
    def NeverStopSAM (self, T_step, detuning_0=-1, Pump_P=2., nmax=1000, abtol=1e-10, reltol=1e-9, out_param='fin_res'):
        self.Pump = self.Pump/abs(self.Pump)
        def deriv_1(dt, field_in):
        # computes the first-order derivative of field_in
            field_fft = np.fft.fft(field_in)
            omega = 2.*np.pi*np.fft.fftfreq(len(field_in),dt)
            out_field = np.fft.ifft(-1j*omega*field_fft)
            return out_field
        
        def deriv_2(dt, field_in):
        # computes the second-order derivative of field_in
            field_fft = np.fft.fft(field_in)
            omega = 2.*np.pi*np.fft.fftfreq(len(field_in),dt)
            field_fft *= -omega**2
            out_field = np.fft.ifft(field_fft)
            return out_field 
        
        def disp(field_in,Dint_in):
        # computes the dispersion term in Fourier space
            field_fft = np.fft.fft(field_in)
            out_field = np.fft.ifft(Dint_in*field_fft)     
            return out_field

        ### define the rhs function
        def LLE_1d(Z, A):
            # for nomalized
            if np.size(self.Dint)==1 and self.Dint == 1:
                 dAdt2 = deriv_2(self.TimeStep, A)
                 dAdT =  1j*dAdt2/2 + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
            elif np.size(self.Dint)==1 and self.Dint == -1:
                 dAdt2 = deriv_2(self.TimeStep, A)
                 dAdT =  -1j*dAdt2/2 + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
            else:  
                # with out raman
                Disp_int = disp(A,self.Dint)
                if self.Traman==0:
                    dAdT =  -1j*Disp_int + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
                else:
                    # with raman
                    dAAdt = deriv_1(self.TimeStep,abs(A)**2)
                    dAdT =  -1j*Disp_int + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A -1j*self.gamma*self.Traman*dAAdt*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump*Pump_P**.5
            return dAdT
        
        r = complex_ode(LLE_1d).set_integrator('dopri5', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        r.set_initial_value(self.seed, 0)# seed the cavity
        
        
        img = mpimg.imread('phase_space.png')
        xx = np.linspace(-1,5,np.size(img,axis=1))
        yy = np.linspace(11,0,np.size(img,axis=0))
        XX,YY = np.meshgrid(xx,yy)
        
        
        fig = plt.figure(figsize=(11,7))        
        plt.subplots_adjust(top=0.95,bottom=0.1,left=0.06,right=0.986,hspace=0.2,wspace=0.16)

        ax1 = plt.subplot(221)
        ax1.pcolormesh(XX,YY,img[:,:,1])
        plt.xlabel('Detuning')
        plt.ylabel('f^2')
        plt.title('Choose the region')
        plt.xlim(min(xx),max(xx))
        dot = plt.plot(detuning_0, Pump_P,'rx')
        
        
        ax2 = plt.subplot(222)
        line, = plt.plot(abs(self.seed)**2)
        plt.ylim(0,1.1)
        plt.ylabel('$|\Psi|^2$')
        
        ax3 = plt.subplot(224)
        line2, = plt.semilogy(self.mu, np.abs(np.fft.fft(self.seed))**2)
        plt.ylabel('PSD')
        plt.xlabel('mode number')
        ### widjets
        axcolor = 'lightgoldenrodyellow'

        resetax = plt.axes([0.4, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Stop', color=axcolor, hovercolor='0.975')
        
        axboxf = plt.axes([0.1, 0.35, 0.1, 0.075])
        text_box_f = TextBox(axboxf, 'f^2', initial=str(Pump_P))
        
        axboxd = plt.axes([0.1, 0.25, 0.1, 0.075])
        text_box_d = TextBox(axboxd, 'Detuning', initial=str(detuning_0))
        
        Run = True
        def setup(event): 
            global Run
            Run = False   
        button.on_clicked(setup)
        
        def onclick(event): 
            if event.inaxes == ax1:
                ix, iy = event.xdata, event.ydata     
                text_box_d.set_val(np.round(ix,4))
                text_box_f.set_val(np.round(iy,4))
                ax1.plot([ix],[iy],'rx')
   

        fig.canvas.mpl_connect('button_press_event', onclick)
        
        while Run:
            dOm_curr = float(text_box_d.text) # get the detuning value
            Pump_P = float(text_box_f.text)
            Field = r.integrate(r.t+T_step)
            F_mod_sq = np.abs(Field)**2
            F_sp = np.abs(np.fft.fft(Field))**2
            line.set_ydata(F_mod_sq)
            line2.set_ydata(F_sp)
            ax2.set_ylim(0, max(F_mod_sq))
            ax3.set_ylim(min(F_sp),max(F_sp))
            plt.pause(1e-10)
    
        
    def Jacobian(self,zeta_0,A,D1):
        
        N = self.N_points
        d2 = self.D2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        index_1 = np.arange(0,N)
        index_2 = np.arange(N,2*N)
        Jacob = np.zeros([2*N+1,2*N+1],dtype=complex)
        
        Jacob[:-1,:-1] += self.LinMatrix(zeta_0,dense=False)
        Jacob[index_1,index_1] += + 2*1j*abs(A[index_1])**2 
        Jacob[index_2,index_2] +=  - 2*1j*abs(A[index_1])**2 
        
        Jacob[index_1,index_2] += 1j*A[index_1]*A[index_1]
        Jacob[index_2,index_1] += -1j*np.conj(A[index_1])*np.conj(A[index_1])
        
        Jacob[index_1,-1] = -(self.D1A(A[index_1]))
        Jacob[index_2,-1] = np.conj(Jacob[index_1,-1])
        
        Jacob[-1,index_1] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        Jacob[-1,index_2] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        
        
        #Jacob_sparse = dia_matrix(Jacob)
        #return Jacob_sparse
        return Jacob
    
    def JacobianForLinAnalysis(self,zeta_0,A):
        
        N = self.N_points
        d2 = self.D2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        index_1 = np.arange(0,N)
        index_2 = np.arange(N,2*N)
        Jacob = np.zeros([2*N,2*N],dtype=complex)
        
        Jacob[:,:] += self.LinMatrix(zeta_0,dense=False)
        Jacob[index_1,index_1] += + 2*1j*abs(A[index_1])**2 
        Jacob[index_2,index_2] +=  - 2*1j*abs(A[index_1])**2 
        
        Jacob[index_1,index_2] += 1j*A[index_1]*A[index_1]
        Jacob[index_2,index_1] += -1j*np.conj(A[index_1])*np.conj(A[index_1])
        
        #Jacob[index_1,-1] = -(self.D1A(A[index_1]))
        #Jacob[index_2,-1] = np.conj(Jacob[index_1,-1])
        
        #Jacob[-1,index_1] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        #Jacob[-1,index_2] = np.real((self.FirstDmat[np.argmax(np.real(A[index_1])),:]))
        
        
        #Jacob_sparse = dia_matrix(Jacob)
        #return Jacob_sparse
        return Jacob
    
    def FirstDerivativeMatrix(self):
        D = np.zeros([self.N_points,self.N_points],dtype=complex)
        index = np.arange(0,self.N_points)
        D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
        D_fourier[index,index] = 1j*self.mu
            
        Fourier_matrix = dft(self.N_points)
        D = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
        
        return D
        
        
    def D1A(self,A):
        D = np.zeros([self.N_points,self.N_points],dtype=complex)
        index = np.arange(0,self.N_points)
        D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
        A_spectrum = np.fft.fft(A)
        D_fourier[index,index] = 1j*self.mu
        #A_spectrum = np.dot(D_fourier,A_spectrum)
        Fourier_matrix = dft(self.N_points)
        D = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
        res = np.dot(D,A)
        #res = np.dot(-1j*self.mu,A_spectrum)#/self.N_points
        #res = np.fft.ifft(A_spectrum)
        
        return res
    
    def DispersionMatrix(self,D1=0,order=0):
        D = np.zeros([self.N_points,self.N_points],dtype=complex)
        index = np.arange(0,self.N_points)
        d2 = self.D2/self.kappa
        #dphi = abs(self.phi[1]-self.phi[0])
        
        if order==0:
            D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
            D_fourier[index,index] = -1j*(self.Dint+D1*self.mu)*2/self.kappa
            
            Fourier_matrix = dft(self.N_points)
            D = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
        
        if order == 2:
            D[index[:-1],index[1:]] = 1j*d2/dphi**2
            D[0,self.N_points-1] =  1j*d2/dphi**2
            D += D.T
            D[index,index]= -2*1j*d2/dphi**2
        if order == 4:
            D[index[:-2],index[2:]] = -1/12*1j*d2/dphi**2
            
            
            D[index[:-1],index[1:]] = 4/3*1j*d2/dphi**2
            
            
            D += D.T
            
            D[0,self.N_points-2] =  -1/12*1j*d2/dphi**2 
            D[self.N_points-1,1] =  -1/12*1j*d2/dphi**2
            
            D[0,self.N_points-1] =  4/3*1j*d2/dphi**2
            D[self.N_points-1,0] =  4/3*1j*d2/dphi**2
            
            D[1,self.N_points-1] =  -1/12*1j*d2/dphi**2 
            D[self.N_points-2,0] =  -1/12*1j*d2/dphi**2
            
            D[index,index]= -5/2*1j*d2/dphi**2
            
        if order == 6:
            D[index[:-3],index[3:]] = 1./90*1j*d2/dphi**2
            
            
            D[index[:-2],index[2:]] = -3./20*1j*d2/dphi**2
            
            
            D[index[:-1],index[1:]] = 3./2*1j*d2/dphi**2
            
            
            D += D.T
            
            D[0,self.N_points-3] =  1./90*1j*d2/dphi**2
            D[self.N_points-1,2] =  1./90*1j*d2/dphi**2
            
            D[0,self.N_points-2] = -3./20*1j*d2/dphi**2
            D[self.N_points-1,1] =  -3./20*1j*d2/dphi**2
            
            D[0,self.N_points-1] =   3./2*1j*d2/dphi**2
            D[self.N_points-1,0] =   3./2*1j*d2/dphi**2
            
            D[1,self.N_points-2] =  1/90*1j*d2/dphi**2 
            D[self.N_points-2,1] =  1/90*1j*d2/dphi**2
            
            D[1,self.N_points-1] = -3./20*1j*d2/dphi**2
            D[self.N_points-2,0] =  -3./20*1j*d2/dphi**2
            
            D[2,self.N_points-1] =  1/90*1j*d2/dphi**2 
            D[self.N_points-3,0] =  1/90*1j*d2/dphi**2
            
            D[index,index]= -49./18*1j*d2/dphi**2
            
        return D
    
    def LinMatrix(self,zeta_0,dense=True):
        self.FirstDmat=self.FirstDerivativeMatrix()
        D = np.zeros([2*self.N_points,2*self.N_points],dtype=complex)
        d2 = self.D2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
      
        D[:self.N_points,:self.N_points] = self.D
        D[self.N_points:,self.N_points:] = np.conj(D[:self.N_points,:self.N_points])
        D[index_1,index_1]+=-(1+ 1j*zeta_0)
        D[index_2,index_2]+=-(1- 1j*zeta_0)
        
        if dense==True:
            D_sparse = dia_matrix(D)
            return D_sparse
        else:
            return D
        
    
    def NewtonRaphson(self,A_input,dOm, Pump,D1=0,HardSeed = True, tol=1e-5,max_iter=50):
        self.D = self.DispersionMatrix(D1=D1,order=0)
        FirstDerivativeMatrix=self.FirstDerivativeMatrix()
        A_guess = np.fft.ifft(A_input)
        
        d2 = self.D2/self.kappa
        zeta_0 = dOm*2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        Aprev = np.zeros(2*self.N_points+1,dtype=complex)
        
        
        f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        f0_direct = np.zeros(Aprev.size-1,dtype=complex)
        f0_direct[index_1] = np.fft.ifft(f0)*self.N_points
        
        f0_direct[index_2] = np.conj(f0_direct[index_1])
        
       
        if HardSeed == False:
            A_guess = A_guess+ f0_direct/(1+1j*zeta_0)
            Aprev[:self.N_points] = A_guess
        else:
            Aprev[:self.N_points] = A_guess*np.sqrt(2*self.g0/self.kappa)
        
        Aprev[index_2] = np.conj(Aprev[:self.N_points])
        Aprev[-1] = D1*2/self.kappa
        
        Ak = np.zeros(Aprev.size,dtype=complex)
        
        

        buf= np.zeros(Aprev.size,dtype=complex)
        buf_prev= np.zeros(Aprev.size,dtype=complex)
        
        M_lin0 = self.LinMatrix(zeta_0)
        
        D1_res=D1*2/self.kappa
       
        print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
        print('xi = ' + str(zeta_0) )
        
        diff = self.N_points
        counter =0
        diff_array=[]
        
        while diff>tol:
            
            
            self.D = self.DispersionMatrix(D1=self.kappa/2*D1_res,order=0)
            J = self.Jacobian(zeta_0, Aprev[index_1],D1=D1_res*self.kappa/2)
            buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
            buf[index_2] = np.conj(buf[index_1])
            #buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]      
            #buf0= buf+  M_lin0.dot(Aprev)+ f0_direct
            buf[:-1] += (self.LinMatrix(zeta_0)).dot(Aprev[:-1]) + f0_direct
            buf[-1]=np.real(self.D1A(np.real(Aprev[index_1]))[np.argmax(np.real(Aprev[index_1]))])
            
            
            Ak = Aprev - np.linalg.solve(J,buf)
            
            D1_res= np.real(Ak[-1])
            
            
            diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
            #print(diff, abs((Ak[-1]-Aprev[-1])/D1_res))
            diff_array += [diff]
            Aprev[:] = Ak[:]
            buf_prev[:]=buf[:]
            Aprev[index_2] = np.conj(Aprev[index_1])
            counter +=1
            
            #plt.scatter(counter,diff,c='k')
            if counter>max_iter:
                print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                res = np.zeros(self.N_points,dtype=complex)
                res = Ak[index_1]
                v = self.kappa/2*D1_res
                return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa), v, diff_array
                break
        print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
        res = np.zeros(self.N_points,dtype=complex)
        res = Ak[index_1]
        v = self.kappa/2*D1_res
        print('D1_res ', D1_res)
        return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa), v,diff_array
    
    def NewtonRaphsonFixedD1(self,A_input,dOm, Pump,HardSeed = True, tol=1e-5,max_iter=50):
        self.D = self.DispersionMatrix(D1=0,order=0)
        
        A_guess = np.fft.ifft(A_input)
        
        d2 = self.D2/self.kappa
        zeta_0 = dOm*2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        Aprev = np.zeros(2*self.N_points,dtype=complex)
        
        
        f0 = pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa**3)
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
        f0_direct = np.zeros(Aprev.size,dtype=complex)
        f0_direct[index_1] = np.fft.ifft(f0)*self.N_points
        
        f0_direct[index_2] = np.conj(f0_direct[index_1])
        
       
        if HardSeed == False:
            A_guess = A_guess+ f0_direct/(1+1j*zeta_0)
            Aprev[:self.N_points] = A_guess
        else:
            Aprev[:self.N_points] = A_guess*np.sqrt(2*self.g0/self.kappa)
        
        Aprev[index_2] = np.conj(Aprev[:self.N_points])
        
        
        Ak = np.zeros(Aprev.size,dtype=complex)
        
        

        buf= np.zeros(Aprev.size,dtype=complex)
        buf_prev= np.zeros(Aprev.size,dtype=complex)
        
        M_lin0 = self.LinMatrix(zeta_0)
        
       
       
        print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
        print('xi = ' + str(zeta_0) )
        
        diff = self.N_points
        counter =0
        diff_array=[]
        
        while diff>tol:
            
            
            #self.D = self.DispersionMatrix(D1=self.kappa/2*D1_res,order=0)
            J = self.JacobianForLinAnalysis(zeta_0, Aprev[index_1])
            buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
            buf[index_2] = np.conj(buf[index_1])
            #buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]      
            #buf0= buf+  M_lin0.dot(Aprev)+ f0_direct
            buf[:] += (self.LinMatrix(zeta_0)).dot(Aprev[:]) + f0_direct
            
            
            
            Ak = Aprev - np.linalg.solve(J,buf)
            
            
            
            
            diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
            #print(diff, abs((Ak[-1]-Aprev[-1])/D1_res))
            diff_array += [diff]
            Aprev[:] = Ak[:]
            buf_prev[:]=buf[:]
            Aprev[index_2] = np.conj(Aprev[index_1])
            counter +=1
            
            #plt.scatter(counter,diff,c='k')
            if counter>max_iter:
                print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                res = np.zeros(self.N_points,dtype=complex)
                res = Ak[index_1]
            
                return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa), diff_array
                break
        print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
        res = np.zeros(self.N_points,dtype=complex)
        res = Ak[index_1]
        
        
        return np.fft.fft(res)/np.sqrt(2*self.g0/self.kappa),diff_array

        
    def LinearStability(self,solution,dOm,v=0,plot_eigvals=True):
        self.D = self.DispersionMatrix(D1=v,order=0)
        A=np.fft.ifft(solution)
        
        d2 = self.D2/self.kappa
        d1 = v*2/self.kappa
        zeta_0 = dOm*2/self.kappa
        dphi = abs(self.phi[1]-self.phi[0])
        field = np.zeros_like(A)
        field = A*np.sqrt(2*self.g0/self.kappa)
        
        
        index_1 = np.arange(0,self.N_points)
        index_2 = np.arange(self.N_points,2*self.N_points)
        
       
        #Full_Matrix=self.Jacobian(zeta_0,field,D1=d1).todense()
        #Full_Matrix=self.Jacobian(zeta_0,field,D1=d1)
        Full_Matrix=self.JacobianForLinAnalysis(zeta_0,field)
        
        
        eig_vals,eig_vec = np.linalg.eig(Full_Matrix)
        
        eigen_vectors = np.zeros([self.N_points,2*self.N_points],dtype=complex)
        if plot_eigvals==True:
            plt.scatter(np.real(eig_vals),np.imag(eig_vals))
            plt.xlabel('Real part')
            plt.ylabel('Imaginary part')
            
        for jj in range(2*self.N_points):
            eigen_vectors[:,jj]=(eig_vec[:self.N_points,jj]).T
            eigen_vectors[:,jj]=np.fft.fft(eigen_vectors[:,jj])
        
        return eig_vals[:-1]*self.kappa/2, eigen_vectors/np.sqrt(2*self.g0/self.kappa)
    def printProgressBar (self, iteration, total, prefix = '', suffix = '', time = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, suffix, time), end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
                print()
 #%%
class SiL_Resonator(Resonator):
    def __init__(self):
        Resonator.__init__(self)
        self.alpha_h = 0 #Linewidth enhancement factor
        self.a = 0 ##differential gain
        self.N0 = 0 #Carrier density at transparency
        self.kappa_laser= 0 #Laser cavity loss rate
        self.kappa_inj = 0 #Laser-microresonator coupling rate
        self.I = 0 #Laser biased current
        self.gamma = 0 #Carrier recombination rate
        self.V = 0 #Volume of active section
        self.eta = 0# Conversion factor
        self.theta = 0 #optical feedback
        self.kappa_sc = 0#cw-ccw coupling rate
        self.zeta = 0 #Hz/A, Current-frequency tuning coefficient
    def Init_From_Dict(self,ResonatorParameters,LaserParameters):
        Resonator.Init_From_Dict(self,ResonatorParameters)
        self.alpha_h = LaserParameters['alpha_h']
        self.a = LaserParameters['a'] ##differential gain
        self.N0 = LaserParameters['N0'] #Carrier density at transparency
        self.kappa_laser= LaserParameters['kappa_laser'] #Laser cavity loss rate
        self.kappa_inj = LaserParameters['kappa_inj'] #Laser-microresonator coupling rate
        self.I = LaserParameters['I'] #Laser biased current
        self.gamma = LaserParameters['gamma'] #Carrier recombination rate
        self.V = LaserParameters['V'] #Volume of active section
        self.eta = LaserParameters['eta']# Conversion factor
        self.theta = LaserParameters['theta']#optical feedback
        self.kappa_sc = ResonatorParameters['kappa_sc']
        self.zeta = LaserParameters['zeta']#Hz/A, Current-frequency tuning coefficient
    def seed_level (self, detuning):
        resonator_field = np.zeros(self.N_points,dtype=complex)
        N = self.N0+self.kappa_laser/self.a/self.V
        E_laser = np.sqrt((-self.I/e/self.V+self.gamma*N)/self.kappa_laser,dtype=complex)
        resonator_field[:] = self.eta*self.kappa_inj*np.exp(self.theta*1j)*E_laser/(self.kappa/2+1j*detuning)
        CCW_amplitude = np.complex_(1j*resonator_field[0]*self.kappa_sc/(self.kappa/2+1j*detuning))
        return resonator_field, CCW_amplitude, E_laser, N
        
    def Propagate_PseudoSpectralSAMCLIB(self, simulation_parameters, dt=5e-4, Seed=[0], CCW_seed=0., E_laser=0., N=0., HardSeed=False):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
      
        
        if HardSeed == False:
            resonator_field, CCW_amplitude, E_laser, N = self.seed_level(detuning[0])
            seed = np.sqrt(2*self.g0/self.kappa)*resonator_field
            CCW_seed  = CCW_amplitude*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
            CCW_seed  = CCW_amplitude*np.sqrt(2*self.g0/self.kappa)
        ### renormalization
        T_rn = (self.kappa/2)*T
        
        f0_direct_sq = np.abs(self.eta*2*self.kappa_inj/self.kappa*np.sqrt(2*self.g0/self.kappa)*E_laser)**2
        print('f0^2 = ' + str(np.round(f0_direct_sq, 2)))
        print('xi [' + str(detuning[0]*2/self.kappa) + ',' +str(detuning[-1]*2/self.kappa)+ ']')
        
        
        t_st = float(T_rn)/len(detuning)
        #dt=1e-4 #t_ph
        
        sol_res = np.ndarray(shape=(len(detuning), self.N_points), dtype='complex') # define an array to store the data
        sol_CCW_res = np.zeros([len(detuning)],dtype=complex)
        sol_Laser = np.zeros([len(detuning)],dtype=complex)
        sol_N = np.zeros([len(detuning)])
        sol_res[0,:] = (seed)/self.N_points
        sol_CCW_res[0] = CCW_seed*self.N_points
        sol_Laser[0] = E_laser#*self.N_points
        sol_N[0] = N#*self.N_points
        print( sol_res[0,0],sol_CCW_res[0], sol_Laser[0] , sol_N[0] )
        
        #%% crtypes defyning
        LLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_lle_core.so')
        
        LLE_core.Propagate_SiL_PseudoSpectralSAM.restype = ctypes.c_void_p
        #LLE_core.Propagate_PseudoSpectralSAM.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.zeros(self.N_points+3,dtype=complex)
        A[:self.N_points] = np.fft.ifft(seed)#*self.N_points
        A[self.N_points] = CCW_seed
        A[self.N_points+1] = E_laser
        A[self.N_points+2] = N
        #plt.plot(abs(A))
        #print(A[0])
        
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_kappa = ctypes.c_double(np.max(self.kappa))
        In_g0 = ctypes.c_double(self.g0)
        
        In_I_laser = ctypes.c_double(self.I)
        In_zeta = ctypes.c_double(self.zeta)
        In_a = ctypes.c_double(self.a)
        In_e = ctypes.c_double(e)
        In_alpha_h = ctypes.c_double(self.alpha_h)
        In_N0 = ctypes.c_double(self.N0)
        In_kappa_laser = ctypes.c_double(self.kappa_laser)
        In_kappa_inj = ctypes.c_double(self.kappa_inj)
        In_gamma = ctypes.c_double(self.gamma)
        In_V = ctypes.c_double(self.V)
        In_eta = ctypes.c_double(self.eta)
        In_theta = ctypes.c_double(self.theta)
        In_kappa_sc = ctypes.c_double(self.kappa_sc)
        
        
        
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
      
        In_det = np.array(2/self.kappa*detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        In_Dint = np.array(self.Dint*2/self.kappa,dtype=ctypes.c_double)
        In_Tmax = ctypes.c_double(T_rn)
        In_Tstep = ctypes.c_double(t_st)
    
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        
            

            
            
            
        In_res_RE = np.zeros(len(detuning)*(self.N_points+3),dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*(self.N_points+3),dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        In_Dint_p = In_Dint.ctypes.data_as(double_p)
        
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        #%%running simulations
        
        LLE_core.Propagate_SiL_PseudoSpectralSAM(In_val_RE_p, In_val_IM_p, In_det_p, In_kappa, In_kappa_laser, In_kappa_sc, In_kappa_inj, In_theta, In_g0, In_alpha_h, In_gamma, In_V, In_a, In_e, In_N0, In_eta, In_I_laser , In_zeta, In_Dint_p, In_Ndet, In_Nt, In_Tmax, In_Tstep, In_dt, In_atol, In_rtol, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        indexes = np.arange(self.N_points+3)
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol_res[ii,ind_modes] = np.fft.fft(In_res_RE[ii*(self.N_points+3)+ind_modes] + 1j*In_res_IM[ii*(self.N_points+3)+ind_modes])
            sol_CCW_res[ii] = In_res_RE[ii*(self.N_points+3)+self.N_points]  + 1j*In_res_IM[ii*(self.N_points+3)+self.N_points]
            sol_Laser[ii] = In_res_RE[ii*(self.N_points+3)+self.N_points+1]  + 1j*In_res_IM[ii*(self.N_points+3)+self.N_points+1]
            sol_N[ii] = In_res_RE[ii*(self.N_points+3)+self.N_points+2]  
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol_res/np.sqrt(2*self.g0/self.kappa), sol_CCW_res/np.sqrt(2*self.g0/self.kappa), sol_Laser, sol_N
        elif out_param == 'fin_res':
            return sol_res[-1, :]/np.sqrt(2*self.g0/self.kappa), sol_CCW_res[-1]/np.sqrt(2*self.g0/self.kappa), sol_Laser[-1], sol_N[-1]
        else:
            print ('wrong parameter')  

#%%              
class CROW(Resonator):#all idenical resonators

        def __init__(self):
        #Physical parameters initialization
            self.n0 = 0
            self.n2 = 0
            self.FSR = 0
            self.w0 = 0
            self.width = 0
            self.height = 0
            self.kappa_0 = 0
            self.Dint = np.array([0])
            
            self.n2t=0
            self.t_th=0
            
            self.Tr = 0
            self.Aeff = 0
            self.Leff = 0
            self.Veff = 0
            self.g0 = 0
            self.gamma = 0
            self.J = np.array([0])
            
            
                
            self.Bus_J = np.array([0])
            self.Bus_Phase = np.array([0])
            self.Snake_coupling=False       
            
            self.Delta = np.array([0])
            self.N_CROW = 0
            self.Delta_D1 = np.zeros(self.N_CROW)
            self.D2 = np.zeros(self.N_CROW)
            self.D3 = np.zeros(self.N_CROW)
            self.kappa_ex =np.array([0])
            self.kappa = self.kappa_0 + self.kappa_ex
            self.N_points = 0
            self.mu = np.array([0])
            self.phi = np.array([0])
            
            self.D2 = np.array([0])
            self.D3 = np.array([0])
            self.D = np.array([0])
            self.M_lin = np.array([0])

            
            
            #self.M_lin = np.array([0])

        def Init_From_Dict(self, resonator_parameters):
        #Physical parameters initialization
            self.n0 = resonator_parameters['n0']
            self.n2 = resonator_parameters['n2']
            self.FSR = resonator_parameters['FSR']
            self.w0 = resonator_parameters['w0']
            self.width = resonator_parameters['width']
            self.height = resonator_parameters['height']
            self.kappa_0 = resonator_parameters['kappa_0']
            self.Dint = resonator_parameters['Dint']
            
            
            self.Tr = 1/self.FSR #round trip time
            self.Aeff = self.width*self.height 
            self.Leff = c/self.n0*self.Tr 
            self.Veff = self.Aeff*self.Leff 
            self.g0 = hbar*self.w0**2*c*self.n2/self.n0**2/self.Veff
            self.gamma = self.n2*self.w0/c/self.Aeff
            self.J = np.array(resonator_parameters['Inter-resonator_coupling'])
            
            #try:
            if 'Snake bus coupling' in resonator_parameters.keys():
                self.Bus_J = np.array(resonator_parameters['Snake bus coupling'])
                self.Bus_Phase = np.array(resonator_parameters['Snake bus phases'])
                self.Snake_coupling= True
            else:
                self.Bus_J = np.array([[0],[0]])
                self.Bus_Phase = np.array([0])
                self.Snake_coupling=False
            if 'T thermal' in resonator_parameters.keys():
                self.n2t = resonator_parameters['n2 thermal']
                self.t_th=resonator_parameters['T thermal']
            
            self.Delta = np.array(resonator_parameters['Resonator detunings'])
            self.N_CROW = len(self.Dint[0,:])
            self.D2 = np.zeros(self.N_CROW)
            self.D3 = np.zeros(self.N_CROW)
            self.kappa_ex = resonator_parameters['kappa_ex']# V
            self.kappa = self.kappa_0 + self.kappa_ex
            self.N_points = len(self.Dint[:,0])
            self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
            
            self.phi = np.linspace(0,2*np.pi,self.N_points)
            if 'Delta D1' in resonator_parameters.keys():
                self.Delta_D1 = resonator_parameters['Delta D1']
            else:
                self.Delta_D1 = np.zeros(self.N_CROW)
            def func(x, a, b, c, d):
                    return a + x*b + c*x**2/2 + d*x**3/6
            for ii in range(0,self.N_CROW):
                self.Dint[:,ii] = np.fft.ifftshift(self.Dint[:,ii])
                
                popt, pcov = curve_fit(func, self.mu, self.Dint[:,ii])
                self.D2[ii] = popt[2]
                self.D3[ii] = popt[3]
            
            ind_phase_modes = np.arange(0,(self.N_CROW-1)*self.N_points)
            ind_phase_modes = ind_phase_modes%self.N_points
            M_lin = diags(-(self.kappa.T.reshape(self.kappa.size)/self.kappa_0+1j*self.Dint.T.reshape(self.Dint.size)*2/self.kappa_0),0) + 1j*diags(self.J[:,:self.N_CROW-1].T.reshape(self.J[:,:self.N_CROW-1].size)*2/self.kappa_0 *np.exp(-1j*ind_phase_modes*np.pi),self.N_points) + 1j*diags(self.J[:,:self.N_CROW-1].T.reshape(self.J[:,:self.N_CROW-1].size)*2/self.kappa_0 *np.exp(1j*ind_phase_modes*np.pi),-self.N_points)
            if self.J[0,:].size == self.N_CROW:
                M_lin+= 1j*diags(self.J[:,self.N_CROW-1].T.reshape(self.J[:,self.N_CROW-1].size)*2/self.kappa_0 *np.exp(-1j*ind_phase_modes[:self.N_points]*np.pi),(self.N_CROW-1)*self.N_points)
                M_lin+= 1j*diags(self.J[:,self.N_CROW-1].T.reshape(self.J[:,self.N_CROW-1].size)*2/self.kappa_0 *np.exp(1j*ind_phase_modes[:self.N_points]*np.pi),-(self.N_CROW-1)*self.N_points)
            
            #self.M_lin = M_lin
            #self.M_lin = M_lin.todense()

            
           
            
        def seed_level (self, pump, detuning):
            
            f_norm = pump*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3)#we pump the first ring
            detuning_norm  = detuning*2/self.kappa_0
            
            #redo
            LinearM = np.eye(self.N_points*self.N_CROW,dtype = complex)
            

            ind_modes = np.arange(self.N_points)
            
            for ii in range(0,self.N_CROW-1):
                LinearM[ind_modes+ii*self.N_points,ind_modes+(ii+1)*self.N_points] = 1j*self.J.T.reshape(self.J.size)[ii*self.N_points +ind_modes]*2/self.kappa_0
            LinearM += LinearM.T
            
            
            indM = np.arange(self.N_points*self.N_CROW)

            LinearM[indM,indM] = -(self.kappa.T.reshape(self.kappa.size)[indM]/self.kappa_0 +1j*self.Delta.T.reshape(self.Delta.size)[indM]/self.kappa_0+ 1j*detuning_norm)
            
            
            
            res_seed = np.zeros_like(f_norm.reshape(f_norm.size))
            res_seed = np.linalg.solve(LinearM,f_norm.T.reshape(f_norm.size))
            res_seed*= 1/np.sqrt(2*self.g0/self.kappa_0)
            #res_seed.reshape((self.N_points,self.N_CROW))
            
            return res_seed
        def noise(self, a):
#        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)
            return a*(np.random.uniform(-1,1,self.N_points*self.N_CROW)+ 1j*np.random.uniform(-1,1,self.N_points*self.N_CROW))
        
        def Linear_analysis(self,plot_dint=True,plot_evec=True):
            M = np.zeros((self.N_CROW,self.N_CROW),dtype='complex')
            ev_arr = np.array([],dtype='complex')
            for ii in range(self.N_points):
                for jj in range(self.N_CROW):
                    M[jj,jj] = 1*self.Dint[ii,jj]+self.mu[ii]*self.Delta_D1[jj] + self.Delta[ii,jj]
                    if jj<self.N_CROW-1:
                        M[jj,jj+1] = self.J[0,jj]
                        M[jj+1,jj] = self.J[0,jj]
                        
                    if self.J[0,:].size==self.N_CROW:
                        M[0,self.N_CROW-1] = self.J[0,self.N_CROW-1]
                        M[self.N_CROW-1,0] = self.J[0,self.N_CROW-1]
                    ev,a = eig(M)
                if self.mu[ii]==0:
                    evec_r = np.real(a.reshape(self.N_CROW**2))
                ev_arr = np.append(ev_arr,ev.T)
            if plot_dint:
                plt.figure()
                for kk in range(self.N_CROW):
                    plt.plot(self.mu,np.real(ev_arr[kk::self.N_CROW]),'k.')
                    plt.xlim(self.mu.min(),self.mu.max())
                    plt.xlabel('Mode number')
                    plt.ylabel('Hybridized D$_{int}$')
                    plt.grid('on')
            if plot_evec:
                fig, ax = plt.subplots()
                patches = []
                for ii in range(self.N_CROW):
                    for jj in range(self.N_CROW):
                        wedge = Wedge((ii*1., jj*1.), .47, 0, 360, width=0.1)
                        patches.append(wedge)
                colors = evec_r
                p = PatchCollection(patches, cmap=cm.seismic,alpha=1)
                p.set_array(np.array(colors))
                ax.add_collection(p)
                fig.colorbar(p, ax=ax)
#                plt.title('J$_0$='+str(self.J/2/np.pi/1e9)+' GHz'+'; Factor='+str(fact)+'; N CROW=' +str(N_crow))
                plt.ylim(-0.5,self.N_CROW*1.-0.5)
                plt.xlim(-0.5,self.N_CROW*1.-0.5)
            return ev_arr
        
        def Propagate_SplitStep(self, simulation_parameters, Pump, Seed=[0], dt=1e-4):
            start_time = time.time()
            T = simulation_parameters['slow_time']
            out_param = simulation_parameters['output']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            #dt = simulation_parameters['time_step']#in photon lifetimes
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa_0)
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3)
            
            print('f0^2 = ' + str(np.round(np.max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
            #dt=1e-4 #t_ph
            
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
           
            self.printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50)
            f0 = np.fft.ifft(f0,axis=0)*self.N_points
            for it in range(1,len(detuning)):
                noise_const = self.noise(eps)
                sol[it-1,:,:] += noise_const.reshape((self.N_points,self.N_CROW))
                self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
                dOm_curr = detuning[it] # detuning value
                t=0
                buf  =  sol[it-1,:,:]
                
                
                buf = np.fft.ifft(buf,axis=0)*self.N_points
               
                while t<t_st:
                    for ii in range(self.N_CROW):
                        #First step
                        buf[:,ii] = np.fft.fft(np.exp(dt*(1j*abs(buf[:,ii])**2 +f0[:,ii]/buf[:,ii]))*buf[:,ii])
                        #second step
                    
                    #buf_vec = np.dot( expm(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0 *np.eye(self.M_lin[:,0].size))),buf.T.reshape(buf.size) )
                    
                    
                    #buf_vec = expm(csc_matrix(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) ))).dot(buf.T.reshape(buf.size))
                    #buf_vec = expm((dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) )).todense()).dot(buf.T.reshape(buf.size))
                    #buf_vec = expm(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) )).dot(buf.T.reshape(buf.size))
                  
                    for ii in range(self.N_CROW):
                        buf[ind_modes,ii] = np.fft.ifft(buf_vec[ii*self.N_points+ind_modes])
                    
                    t+=dt
                sol[it,:,:] = np.fft.fft(buf,axis=0)/len(buf)
                #sol[it,:] = buf
                
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
            
        def Propagate_SAM(self, simulation_parameters, Pump, Seed=[0]):
            start_time = time.time()
            
            T = simulation_parameters['slow_time']
            abtol = simulation_parameters['absolute_tolerance']
            reltol = simulation_parameters['relative_tolerance']
            out_param = simulation_parameters['output']
            nmax = simulation_parameters['max_internal_steps']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            #dt = simulation_parameters['time_step']#in photon lifetimes
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0,0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
            else:
                seed = Seed.T.reshape(Seed.size)*np.sqrt(2*self.g0/self.kappa_0)
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3)
            
            print('f0^2 = ' + str(np.round(np.max(abs(f0)**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
            #dt=1e-4 #t_ph
            
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
           
            self.printProgressBar(0, nn, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
            #def RHS(Time, A):
            #    A = A - noise_const#self.noise(eps)
            #    A_dir = np.zeros(A.size,dtype=complex)
              
            #    for ii in range(self.N_CROW):
            #        A_dir[ii*self.N_points+ind_modes] = np.fft.ifft(A[ii*self.N_points+ind_modes])## in the direct space
            #    A_dir*=self.N_points
            #    dAdT =  (self.M_lin -1j*dOm_curr*2/self.kappa_0* np.eye(self.N_points*self.N_CROW)).dot(A) + f0.reshape(f0.size) 
            #    for ii in range(self.N_CROW):
            #        dAdT[0,ii*self.N_points+ind_modes]+=1j*np.fft.fft(A_dir[ii*self.N_points+ind_modes]*np.abs(A_dir[ii*self.N_points+ind_modes])**2)/self.N_points
            #    return dAdT
            def RHS(Time, A):
                A = A - noise_const#self.noise(eps)
                A_dir = np.zeros(A.size,dtype=complex)
                dAdT = np.zeros(A.size,dtype=complex)
              
                for ii in range(self.N_CROW):
                    A_dir[ii*self.N_points+ind_modes] = np.fft.ifft(A[ii*self.N_points+ind_modes])## in the direct space
                A_dir*=self.N_points
                dAdT =  (-self.kappa.T.reshape(self.kappa.size)/2-1j*self.Dint.T.reshape(self.Dint.size) -1j*dOm_curr)*A*2/self.kappa_0 + f0.reshape(f0.size) 
                dAdT[0*self.N_points+ind_modes] += 1j*self.J[:,0]*2/self.kappa_0 *np.exp(-1j*self.mu*np.pi)*A[1*self.N_points+ind_modes]+1j*np.fft.fft(A_dir[0*self.N_points+ind_modes]*np.abs(A_dir[0*self.N_points+ind_modes])**2)/self.N_points
                dAdT[(self.N_CROW-1)*self.N_points+ind_modes] += 1j*self.J[:,self.N_CROW-2]*2/self.kappa_0 *np.exp(1j*self.mu*np.pi)*A[((self.N_CROW-2))*self.N_points+ind_modes]+1j*np.fft.fft(A_dir[(self.N_CROW-1)*self.N_points+ind_modes]*np.abs(A_dir[(self.N_CROW-1)*self.N_points+ind_modes])**2)/self.N_points
                for ii in range(1,self.N_CROW-1):
                    dAdT[ii*self.N_points+ind_modes]+= 1j*self.J[:,ii]*2/self.kappa_0 *np.exp(-1j*self.mu*np.pi)*A[(ii+1)*self.N_points+ind_modes] + 1j*self.J[:,ii-1]*2/self.kappa_0 *np.exp(1j*self.mu*np.pi)*A[(ii-1)*self.N_points+ind_modes] +  1j*np.fft.fft(A_dir[ii*self.N_points+ind_modes]*np.abs(A_dir[ii*self.N_points+ind_modes])**2)/self.N_points
                return dAdT
            r = complex_ode(RHS).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
            #r = ode(RHS).set_integrator('zvode', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
            
            r.set_initial_value(seed, 0)# seed the cavity
            
            for it in range(1,len(detuning)):
                self.printProgressBar(it + 1, nn, prefix = 'Progress:', suffix = 'Complete,', time='elapsed time = ' + '{:04.1f}'.format(time.time() - start_time) + ' s', length = 50)
                dOm_curr = detuning[it] # detuning value
                res = r.integrate(r.t+t_st)
                for ii in range(self.N_CROW):
                    sol[it,ind_modes,ii] = res[ii*self.N_points+ind_modes]
                
                
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
                
        def Propagate_SAMCLIB(self, simulation_parameters, Pump, BC, Seed=[0], dt=5e-4,HardSeed=False):
            
            
            T = simulation_parameters['slow_time']
            abtol = simulation_parameters['absolute_tolerance']
            reltol = simulation_parameters['relative_tolerance']
            out_param = simulation_parameters['output']
            nmax = simulation_parameters['max_internal_steps']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            
            
            if HardSeed == False:
                
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
                
            else:
                seed = Seed.T.reshape(Seed.size)*np.sqrt(2*self.g0/self.kappa_0)
                seed/=self.N_points
            
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3),axis=0)*self.N_points
            
            print('f0^2 = ' + str(np.round((abs(f0[0,:])**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
                
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            j = np.zeros(self.J[0,:].size)
            delta = np.zeros(self.Delta[0,:].size)
            kappa = np.zeros(self.N_CROW)
            
            
            for ii in range(self.J[0,:].size):
                j[ii] = self.J[0,ii]
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
                kappa[ii] = self.kappa[0,ii]
                delta[ii] = self.Delta[0,ii]
            
            #if self.Snake_coupling==True:
            bus_j = np.zeros(self.Bus_J[0,:].size)
            bus_phase = np.zeros(self.Bus_Phase[:].size)
            for ii in range(self.Bus_J[0,:].size):
                bus_j[ii] = self.Bus_J[0,ii]
                
            for ii in range(self.Bus_Phase[:].size):
                bus_phase[ii] = self.Bus_Phase[ii]
            
            f0 =(f0.T.reshape(f0.size))
            #%% crtypes definition
            
            if self.J[0,:].size == self.N_CROW:
                BC='PERIODIC'
            elif self.J[0,:].size == self.N_CROW-1:
                BC='OPEN'
            else:
                sys.exit('Unkown type of CROW')
                    
            
            if BC=='OPEN':
                if self.Snake_coupling==False:
                    if abs(self.Delta_D1.max())==0:
                        CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_crow_core.so')
                    if abs(self.Delta_D1.max())>0:
                        CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_crow_core_different_FSR.so')
                else :
                    CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_snake_coupling_crow_core.so')    
            elif BC=='PERIODIC':
                if self.Snake_coupling==False:
                    CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_periodic_crow_core.so')
                else:
                    CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_snake_coupling_periodic_crow_core.so')    
            else:
                sys.exit('Solver has not been found')
            
            if self.n2t==0:
                CROW_core.PropagateSAM.restype = ctypes.c_void_p
            else:
                CROW_core.PropagateThermalSAM.restype = ctypes.c_void_p
                
            A = np.zeros([self.N_CROW*self.N_points],dtype=complex)
            for ii in range(self.N_CROW):    
                A[ii*self.N_points+ind_modes] = np.fft.ifft(seed[ii*self.N_points+ind_modes])*self.N_points
                
        
            In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
            In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
            In_phi = np.array(self.phi,dtype=ctypes.c_double)
            In_Nphi = ctypes.c_int(self.N_points)
            In_Ncrow = ctypes.c_int(self.N_CROW)
            In_f_RE = np.array(np.real(f0 ),dtype=ctypes.c_double)
            In_f_IM = np.array(np.imag(f0 ),dtype=ctypes.c_double)
            In_atol = ctypes.c_double(abtol)
            In_rtol = ctypes.c_double(reltol)
            
            In_det = np.array(detuning,dtype=ctypes.c_double)
            In_Ndet = ctypes.c_int(len(detuning))
            In_D2 = np.array(self.D2,dtype=ctypes.c_double)
            
            if self.n2t!=0:
                In_t_th = ctypes.c_double(self.t_th)
                In_n2 = ctypes.c_double(self.n2)
                In_n2t = ctypes.c_double(self.n2t)
            
            In_kappa = np.array(kappa,dtype=ctypes.c_double)
            In_delta = np.array(delta,dtype=ctypes.c_double)
            In_kappa_0 = ctypes.c_double(self.kappa_0)
            In_J = np.array(j,dtype=ctypes.c_double)
            
            #if self.Snake_coupling==True:
            In_bus_J = np.array(bus_j,dtype=ctypes.c_double)
            In_bus_phase = np.array(bus_phase,dtype=ctypes.c_double)
            
            In_Tmax = ctypes.c_double(t_st)
            In_Nt = ctypes.c_int(int(t_st/dt)+1)
            In_dt = ctypes.c_double(dt)
            In_noise_amp = ctypes.c_double(eps)
            
            In_res_RE = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            In_res_IM = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            
            double_p=ctypes.POINTER(ctypes.c_double)
            
            if self.Delta_D1.size==self.N_CROW:
                In_delta_D1 = np.array(self.Delta_D1,dtype=ctypes.c_double)
                In_delta_D1_p = In_delta_D1.ctypes.data_as(double_p)
            
            In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
            In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
            In_phi_p = In_phi.ctypes.data_as(double_p)
            In_det_p = In_det.ctypes.data_as(double_p)
            In_D2_p = In_D2.ctypes.data_as(double_p)
            
            In_kappa_p = In_kappa.ctypes.data_as(double_p)
            In_delta_p = In_delta.ctypes.data_as(double_p)
            In_J_p = In_J.ctypes.data_as(double_p)
            
            #if self.Snake_coupling==True:
            In_bus_j_p = In_bus_J.ctypes.data_as(double_p)
            In_bus_phase_p = In_bus_phase.ctypes.data_as(double_p)
            
            In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
            In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
            
            In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
            In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
            
            
            
                
            if self.Snake_coupling==False:
                
                if self.n2t==0:
                    if abs(self.Delta_D1.max())==0:
                        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
                    else:
                        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_delta_D1_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
                else:
                    CROW_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_t_th, In_n2, In_n2t, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            else:
                CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_bus_j_p, In_bus_phase_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
            ind_modes = np.arange(self.N_points)
            for ii in range(0,len(detuning)):
                for jj in range(self.N_CROW):
                    sol[ii,ind_modes,jj] = np.fft.fft(In_res_RE[ii*self.N_points*self.N_CROW + jj*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points*self.N_CROW+ jj*self.N_points+ind_modes])#/np.sqrt(self.N_points)
                
            #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                        
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
         
        def Propagate_PSEUDO_SPECTRAL_SAMCLIB(self, simulation_parameters, Pump, BC, Seed=[0], dt=5e-4,HardSeed=False):
            
            
            T = simulation_parameters['slow_time']
            abtol = simulation_parameters['absolute_tolerance']
            reltol = simulation_parameters['relative_tolerance']
            out_param = simulation_parameters['output']
            nmax = simulation_parameters['max_internal_steps']
            detuning = simulation_parameters['detuning_array']
            eps = simulation_parameters['noise_level']
            
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            
            
            if HardSeed == False:
                
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa_0)
                
            else:
                seed = Seed.T.reshape(Seed.size)*np.sqrt(2*self.g0/self.kappa_0)
                seed/=self.N_points
            
            ### renormalization
            T_rn = (self.kappa_0/2)*T
            f0 = np.fft.ifft(pump*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3),axis=0)*self.N_points
            
            print('f0^2 = ' + str(np.round((abs(f0[0,:])**2), 2)))
            print('xi [' + str(detuning[0]*2/self.kappa_0) + ',' +str(detuning[-1]*2/self.kappa_0)+ '] (normalized on ' r'$kappa_0/2)$')
            noise_const = self.noise(eps) # set the noise level
            nn = len(detuning)
            
            t_st = float(T_rn)/len(detuning)
                
            sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_CROW), dtype='complex') # define an array to store the data
            
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            j = np.zeros(self.J[0,:].size)
            delta = np.zeros(self.Delta[0,:].size)
            kappa = np.zeros(self.N_CROW)
            
            
            for ii in range(self.J[0,:].size):
                j[ii] = self.J[0,ii]
            for ii in range(self.N_CROW):
                sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
                kappa[ii] = self.kappa[0,ii]
                delta[ii] = self.Delta[0,ii]
            
            #if self.Snake_coupling==True:
            bus_j = np.zeros(self.Bus_J[0,:].size)
            bus_phase = np.zeros(self.Bus_Phase[:].size)
            for ii in range(self.Bus_J[0,:].size):
                bus_j[ii] = self.Bus_J[0,ii]
                
            for ii in range(self.Bus_Phase[:].size):
                bus_phase[ii] = self.Bus_Phase[ii]
            
            f0 =(f0.T.reshape(f0.size))
            #%% crtypes definition
            
            if self.J[0,:].size == self.N_CROW:
                BC='PERIODIC'
            elif self.J[0,:].size == self.N_CROW-1:
                BC='OPEN'
            else:
                sys.exit('Unkown type of CROW')
                    
            
            if BC=='OPEN':
                CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_crow_core.so')   
            
            elif BC=='PERIODIC':
                CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_periodic_crow_core.so')
            else:
                sys.exit('Solver has not been found')
            print(BC)
            CROW_core.Propagate_PseudoSpectralSAM.restype = ctypes.c_void_p
            #if self.n2t==0:
            #    
            #else:
            #    pass
                #CROW_core.PropagateThermalSAM.restype = ctypes.c_void_p
                
            A = np.zeros([self.N_CROW*self.N_points],dtype=complex)
            for ii in range(self.N_CROW):    
                A[ii*self.N_points+ind_modes] = np.fft.ifft(seed[ii*self.N_points+ind_modes])*self.N_points
                
        
            In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
            In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
            In_phi = np.array(self.phi,dtype=ctypes.c_double)
            In_Nphi = ctypes.c_int(self.N_points)
            In_Ncrow = ctypes.c_int(self.N_CROW)
            In_f_RE = np.array(np.real(f0 ),dtype=ctypes.c_double)
            In_f_IM = np.array(np.imag(f0 ),dtype=ctypes.c_double)
            In_atol = ctypes.c_double(abtol)
            In_rtol = ctypes.c_double(reltol)
            
            In_det = np.array(detuning,dtype=ctypes.c_double)
            In_Ndet = ctypes.c_int(len(detuning))
            In_Dint = np.array(self.Dint.T.reshape(self.Dint.size),dtype=ctypes.c_double)
            
            if self.n2t!=0:
                In_t_th = ctypes.c_double(self.t_th)
                In_n2 = ctypes.c_double(self.n2)
                In_n2t = ctypes.c_double(self.n2t)
            
            In_kappa = np.array(kappa,dtype=ctypes.c_double)
            In_delta = np.array(delta,dtype=ctypes.c_double)
            In_kappa_0 = ctypes.c_double(self.kappa_0)
            In_J = np.array(j,dtype=ctypes.c_double)
            
            #if self.Snake_coupling==True:
            In_bus_J = np.array(bus_j,dtype=ctypes.c_double)
            In_bus_phase = np.array(bus_phase,dtype=ctypes.c_double)
            
            In_Tmax = ctypes.c_double(t_st)
            In_Nt = ctypes.c_int(int(t_st/dt)+1)
            In_dt = ctypes.c_double(dt)
            In_noise_amp = ctypes.c_double(eps)
            
            In_res_RE = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            In_res_IM = np.zeros(len(detuning)*self.N_points*self.N_CROW,dtype=ctypes.c_double)
            
            double_p=ctypes.POINTER(ctypes.c_double)
            
            if self.Delta_D1.size==self.N_CROW:
                In_delta_D1 = np.array(self.Delta_D1,dtype=ctypes.c_double)
                In_delta_D1_p = In_delta_D1.ctypes.data_as(double_p)
            
            In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
            In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
            In_phi_p = In_phi.ctypes.data_as(double_p)
            In_det_p = In_det.ctypes.data_as(double_p)
            In_Dint_p = In_Dint.ctypes.data_as(double_p)
            
            In_kappa_p = In_kappa.ctypes.data_as(double_p)
            In_delta_p = In_delta.ctypes.data_as(double_p)
            In_J_p = In_J.ctypes.data_as(double_p)
            
            #if self.Snake_coupling==True:
            In_bus_j_p = In_bus_J.ctypes.data_as(double_p)
            In_bus_phase_p = In_bus_phase.ctypes.data_as(double_p)
            
            In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
            In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
            
            In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
            In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
            
   
                
            if self.n2t==0:
                CROW_core.Propagate_PseudoSpectralSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)                
                
            else:
                CROW_core.Propagate_PseudoSpectralThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_t_th, In_n2, In_n2t, In_delta_p, In_J_p, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)                
            #if self.n2t==0:
            #    if abs(self.Delta_D1.max())==0:
            #        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            #    else:
            #        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_delta_p, In_delta_D1_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            #else:
            #    CROW_core.PropagateThermalSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_kappa_p, In_kappa_0, In_t_th, In_n2, In_n2t, In_delta_p, In_J_p, In_phi_p, In_D2_p, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ncrow, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
            ind_modes = np.arange(self.N_points)
            for ii in range(0,len(detuning)):
                for jj in range(self.N_CROW):
                    sol[ii,ind_modes,jj] = np.fft.fft(In_res_RE[ii*self.N_points*self.N_CROW + jj*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points*self.N_CROW+ jj*self.N_points+ind_modes])#/np.sqrt(self.N_points)
                
            #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                        
            if out_param == 'map':
                return sol/np.sqrt(2*self.g0/self.kappa_0)
            elif out_param == 'fin_res':
                return sol[-1, :]/np.sqrt(2*self.g0/self.kappa_0)
            else:
                print ('wrong parameter')
            
        def Jacobian(self,j,d2,dphi,delta, kappa, zeta_0,A):
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            J = np.zeros([2*N_m*N_res,2*N_m*N_res],dtype=complex)
            
            
            
            for jj in ind_res:
                            
                J[jj*N_m+ind_m,jj*N_m+ind_m] =0.5*(-(kappa[jj]+ 1j*(zeta_0+delta[jj]))  - 2*1j*d2[jj]/dphi**2 +2*1j*abs(A[jj*N_m+ind_m])**2 )
                J[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 0.5*(-(kappa[jj]- 1j*(zeta_0+delta[jj]))  + 2*1j*d2[jj]/dphi**2 -2*1j*abs(A[jj*N_m+ind_m])**2)
                
                J[jj*N_m+ind_m[:-1],jj*N_m+ind_m[1:]] = 1j*d2[jj]/dphi**2
                J[jj*N_m+0,jj*N_m+N_m-1] =  1j*d2[jj]/dphi**2
                
                J[(jj+N_res)*N_m+ind_m[:-1],(jj+N_res)*N_m+ind_m[1:]] = -1j*d2[jj]/dphi**2
                J[(jj+N_res)*N_m+0,(jj+N_res)*N_m+N_m-1] =  -1j*d2[jj]/dphi**2
           
            for jj in ind_res[:-1]:
                 J[jj*N_m+ind_m[:],(jj+1)*N_m+ind_m[:]] = 1j*j[jj]
                 J[(jj+N_res)*N_m+ind_m[:],((jj+N_res)+1)*N_m+ind_m[:]] = -1j*j[jj]
                
                
            if d2.size==j.size:
                jj=d2.size-1
                J[jj*N_m+ind_m[:],(0)*N_m+ind_m[:]] = 1j*j[jj]
                J[(jj+N_res)*N_m+ind_m[:],(0+N_res)*N_m+ind_m[:]] = -1j*j[jj]
                #D[jj*N_m*2+N_m+ind_m[:],(0)*N_m*2+N_m+ind_m[:]] = -1j*j[jj]
            J += J.T     
            for jj in ind_res:
                 J[jj*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 1j*A[jj*N_m+ind_m]*A[jj*N_m+ind_m]
                 J[(jj+N_res)*N_m+ind_m,(jj)*N_m+ind_m] = -1j*np.conj(A[jj*N_m+ind_m])*np.conj(A[jj*N_m+ind_m])
                        
            #J+=np.conj(J.T)
            #for jj in ind_res:
                            
            #    J[jj*N_m+ind_m,jj*N_m+ind_m] = 2*1j*abs(A[jj*N_m+ind_m])**2
            #    J[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] = -2*1j*abs(A[jj*N_m+ind_m])**2
            
            #J += M_lin
            Jacob_sparse = dia_matrix(J)
            return Jacob_sparse
        def JacobianMatrix(self,zeta_0,A,order=0):
            if self.M_lin[:].size==1:
                self.M_lin = self.LinearMatrix(zeta_0,order)
                
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            J = np.zeros([2*N_m*N_res,2*N_m*N_res],dtype=complex)
            for jj in ind_res:
                J[jj*N_m+ind_m,jj*N_m+ind_m] += +2*1j*abs(A[jj*N_m+ind_m])**2 
                J[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] += -2*1j*abs(A[jj*N_m+ind_m])**2
                J[jj*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 1j*A[jj*N_m+ind_m]*A[jj*N_m+ind_m]
                J[(jj+N_res)*N_m+ind_m,(jj)*N_m+ind_m] = -1j*np.conj(A[jj*N_m+ind_m])*np.conj(A[jj*N_m+ind_m])
            J+= self.M_lin
            
            
            if order == 0:
                Jacob_sparse =(J) 
            else:
                Jacob_sparse = dia_matrix(J)    
                
            return Jacob_sparse
        def DispersionMatrix(self,order=0):
            
            
            D = np.zeros([self.N_points*self.N_CROW,self.N_points*self.N_CROW],dtype=complex)
           
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            d2 = self.D2/self.kappa_0
            dphi = abs(self.phi[1]-self.phi[0])
            if order == 0:
                Fourier_matrix = dft(self.N_points)
                D_fourier = np.zeros([self.N_points,self.N_points],dtype=complex)
               
                for jj in ind_res:      
                   D_fourier[ind_m,ind_m] =-(self.kappa[:,jj]/2+1j*self.Delta[:,jj])*2/self.kappa_0 -1j*self.Dint[:,jj]*2/self.kappa_0
                   D[jj*self.N_points+ind_m[0]:jj*self.N_points+ind_m[-1]+1,jj*self.N_points+ind_m[0]:jj*self.N_points+ind_m[-1]+1] = np.dot(np.dot(Fourier_matrix,D_fourier),np.conj(Fourier_matrix.T)/self.N_points)
            if order == 2:
                N_m = self.N_points
                N_res = self.N_CROW
                for jj in ind_res:
                    D[jj*N_m+ind_m[:-1],jj*N_m+ind_m[1:]] = 1j*d2[jj]/dphi**2
                    D[jj*N_m+0,jj*N_m+N_m-1] =  1j*d2[jj]/dphi**2
                    
                    #D[(jj+N_res)*N_m+ind_m[:-1],(jj+N_res)*N_m+ind_m[1:]] = -1j*d2[jj]/dphi**2
                    #D[(jj+N_res)*N_m+0,(jj+N_res)*N_m+N_m-1] =  -1j*d2[jj]/dphi**2
                    
                    
                    D[jj*N_m+ind_m,jj*N_m+ind_m] = 0.5*(-(self.kappa[:,jj]/2+1j*self.Delta[:,jj])*2/self.kappa_0  - 2*1j*d2[jj]/dphi**2)
                D+=D.T
           
            return D
        
        def LinearMatrix(self,zeta_0,order=0):
            
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            if self.D[:].size==1:
                self.D = self.DispersionMatrix(order)
            M_lin = np.zeros([2*self.N_points*self.N_CROW,2*self.N_points*self.N_CROW],dtype=complex) 
            
            for jj in ind_res[:-1]:
                M_lin[jj*N_m+ind_m[:],(jj+1)*N_m+ind_m[:]] += 1j*self.J[0,jj]*2/self.kappa_0
                M_lin[(jj+N_res)*N_m+ind_m[:],((jj+N_res)+1)*N_m+ind_m[:]] += -1j*self.J[0,jj]*2/self.kappa_0
            if self.Dint[0,:].size==self.J[0,:].size:
                jj=self.Dint[0,:].size-1
                M_lin[jj*N_m+ind_m[:],(0)*N_m+ind_m[:]] += 1j*self.J[0,jj]*2/self.kappa_0
                M_lin[(jj+N_res)*N_m+ind_m[:],(0+N_res)*N_m+ind_m[:]] += -1j*self.J[0,jj]*2/self.kappa_0
                
            M_lin += M_lin.T
            
            
            M_lin[:self.N_points*self.N_CROW,:self.N_points*self.N_CROW] += self.D
            M_lin[self.N_points*self.N_CROW:,self.N_points*self.N_CROW:] += np.conj(self.D)
            
            for jj in ind_res:
                M_lin[jj*N_m+ind_m,jj*N_m+ind_m] += -1j*(zeta_0) 
                M_lin[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] += 1j*(zeta_0) 
            return M_lin
        def LinMatrix(self,j,d2,dphi,delta, kappa, zeta_0):
            
            N_m = self.N_points
            N_res = self.N_CROW
            ind_m = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            
            D = np.zeros([2*N_m*N_res,2*N_m*N_res],dtype=complex)
            
            for jj in ind_res:
                D[jj*N_m+ind_m[:-1],jj*N_m+ind_m[1:]] = 1j*d2[jj]/dphi**2
                D[jj*N_m+0,jj*N_m+N_m-1] =  1j*d2[jj]/dphi**2
                
                D[(jj+N_res)*N_m+ind_m[:-1],(jj+N_res)*N_m+ind_m[1:]] = -1j*d2[jj]/dphi**2
                D[(jj+N_res)*N_m+0,(jj+N_res)*N_m+N_m-1] =  -1j*d2[jj]/dphi**2
                
                
                D[jj*N_m+ind_m,jj*N_m+ind_m] = 0.5*(-(kappa[jj]+ 1j*(zeta_0+delta[jj]))  - 2*1j*d2[jj]/dphi**2)
                D[(jj+N_res)*N_m+ind_m,(jj+N_res)*N_m+ind_m] = 0.5*(-(kappa[jj]- 1j*(zeta_0+delta[jj]))  + 2*1j*d2[jj]/dphi**2)
                
                
            for jj in ind_res[:-1]:
                D[jj*N_m+ind_m[:],(jj+1)*N_m+ind_m[:]] = 1j*j[jj]
                D[(jj+N_res)*N_m+ind_m[:],((jj+N_res)+1)*N_m+ind_m[:]] = -1j*j[jj]
                
                
            if d2.size==j.size:
                jj=d2.size-1
                D[jj*N_m+ind_m[:],(0)*N_m+ind_m[:]] = 1j*j[jj]
                D[(jj+N_res)*N_m+ind_m[:],(0+N_res)*N_m+ind_m[:]] = -1j*j[jj]
                #D[jj*N_m*2+N_m+ind_m[:],(0)*N_m*2+N_m+ind_m[:]] = -1j*j[jj]
            D += D.T
            
            #D[:N_m*N_res]=np.conj(D[N_m*N_res:])
            
            return D
        def NewtonRaphsonDirectSpace(self,Seed_sol,dOm, Pump, HardSeed = True, tol=1e-5,max_iter=50,order=0):
            A_guess = np.fft.ifft(Seed_sol,axis=0)
            result = np.zeros_like(A_guess,dtype=complex)
            zeta_0 = dOm*2/self.kappa_0
            N_m = self.N_points
            N_res = self.N_CROW
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            f0_direct = np.fft.ifft(pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3),axis=0)*self.N_points
            index_1 = np.arange(0,N_m*N_res)
            index_2 = np.arange(N_m*N_res,2*N_m*N_res)
            
            f0_direct =(f0_direct.T.reshape(f0_direct.size))
            self.D = self.DispersionMatrix(order=order)
            self.M_lin = self.LinearMatrix(zeta_0)
            
            
            Aprev = np.zeros(2*N_m*N_res,dtype=complex)
            if HardSeed == False:
                A_guess = A_guess.T.reshape(A_guess.size)+ solve(Mlin[:N_m*N_res,:N_m*N_res],-f0_direct)
                Aprev[index_1] = A_guess
            else:
                Aprev[index_1] = A_guess.T.reshape(A_guess.size)*np.sqrt(2*self.g0/self.kappa_0)
            
            
            Aprev[index_2] = np.conj(Aprev[index_1])
            
            Ak = np.zeros(Aprev.size,dtype=complex)
            
            
            
            f0 = np.zeros(Aprev.size,dtype=complex)
            f0[index_1] = f0_direct
            
            f0[index_2] = np.conj(f0[index_1])
    
            buf= np.zeros(Aprev.size,dtype=complex)
            J=self.JacobianMatrix(zeta_0, Aprev[index_1],order)
            print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
            print('xi = ' + str(zeta_0) )
            
            diff = self.N_points
            counter =0
            diff_array=[]
            isSparse = isspmatrix(J)
            while diff>tol:
                J=self.JacobianMatrix(zeta_0, Aprev[index_1],order)
                buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
                #buf[index_2] = np.conj(buf[index_1])
                buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]         
                buf += (self.M_lin).dot(Aprev) + f0
                
                if isSparse==False:
                    Ak = Aprev - solve_dense(J,buf)
                else:
                    Ak = Aprev - solve_sparse(J,buf)
                
                
                diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
                diff_array += [diff]

                
                Aprev = Ak
                Aprev[index_2] = np.conj(Aprev[index_1])
                counter +=1
                print(diff)
                #plt.scatter(counter,diff,c='k')
                if counter>max_iter:
                    print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                    res = np.zeros(self.N_points,dtype=complex)
                    res = Ak[index_1]
                    
                    for jj in ind_res:
                        result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
                    return result/np.sqrt(2*self.g0/self.kappa_0), diff_array
                    break
            print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
            res = np.zeros(self.N_points,dtype=complex)
            res = Ak[index_1]
            for jj in ind_res:
                result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
            return result/np.sqrt(2*self.g0/self.kappa_0), diff_array    
            
        def NewtonRaphson(self,Seed_sol,dOm, Pump, HardSeed = True, tol=1e-5,max_iter=50):
            A_guess = np.fft.ifft(Seed_sol,axis=0)
            result = np.zeros_like(A_guess,dtype=complex)
            N_m = self.N_points
            N_res = self.N_CROW
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            j = np.zeros(self.J[0,:].size)
            delta = np.zeros(self.Delta[0,:].size)
            kappa = np.zeros(self.N_CROW)
            d2 = np.zeros(self.N_CROW)
            
            for ii in range(self.J[0,:].size):
                j[ii] = self.J[0,ii]*2/self.kappa_0
            for ii in range(self.N_CROW):
                #A_guess[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
                kappa[ii] = self.kappa[0,ii]/self.kappa_0
                delta[ii] = self.Delta[0,ii]*2/self.kappa_0
                d2[ii] = self.D2[ii]/self.kappa_0
                
            
            
            zeta_0 = dOm*2/self.kappa_0
            
            
            
            dphi = abs(self.phi[1]-self.phi[0])
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            
            f0_direct = np.fft.ifft(pump*np.sqrt(8*self.g0*np.max(self.kappa_ex)/self.kappa_0**3),axis=0)*self.N_points
            
            index_1 = np.arange(0,N_m*N_res)
            index_2 = np.arange(N_m*N_res,2*N_m*N_res)
            
            f0_direct =(f0_direct.T.reshape(f0_direct.size))
            
            M_lin = self.LinMatrix(j, d2, dphi, delta, kappa, zeta_0)
            
            
            Aprev = np.zeros(2*N_m*N_res,dtype=complex)
            if HardSeed == False:
                A_guess = A_guess.T.reshape(A_guess.size)+ solve(Mlin[:N_m*N_res,:N_m*N_res],-f0_direct)
                Aprev[index_1] = A_guess
            else:
                Aprev[index_1] = A_guess.T.reshape(A_guess.size)*np.sqrt(2*self.g0/self.kappa_0)
            
            
            Aprev[index_2] = np.conj(Aprev[index_1])
            
            Ak = np.zeros(Aprev.size,dtype=complex)
            
            
            
            f0 = np.zeros(Aprev.size,dtype=complex)
            f0[index_1] = f0_direct
            
            f0[index_2] = np.conj(f0[index_1])
    
            buf= np.zeros(Aprev.size,dtype=complex)
            J=self.Jacobian(j, d2, dphi, delta, kappa, zeta_0, Aprev[index_1])
            print('f0^2 = ' + str(np.round(max(abs(f0_direct)**2), 2)))
            print('xi = ' + str(zeta_0) )
            
            diff = self.N_points
            counter =0
            diff_array=[]
            
            while diff>tol:
                J=self.Jacobian(j, d2, dphi, delta, kappa, zeta_0, Aprev[index_1])
                
                buf[index_1] =  1j*abs(Aprev[index_1])**2*Aprev[index_1]         
                #buf[index_2] = np.conj(buf[index_1])
                buf[index_2] =  -1j*abs(Aprev[index_2])**2*Aprev[index_2]         
                buf += (M_lin).dot(Aprev) + f0
                
                #inv(M_lin)
                Ak = Aprev - solve_sparse(J,buf)
                
                diff = np.sqrt(abs((Ak-Aprev).dot(np.conj(Ak-Aprev))/(Ak.dot(np.conj(Ak)))))
                diff_array += [diff]

                
                Aprev = Ak
                Aprev[index_2] = np.conj(Aprev[index_1])
                counter +=1
                print(diff)
                #plt.scatter(counter,diff,c='k')
                if counter>max_iter:
                    print("Did not coverge in " + str(max_iter)+ " iterations, relative error is " + str(diff))
                    res = np.zeros(self.N_points,dtype=complex)
                    res = Ak[index_1]
                    
                    for jj in ind_res:
                        result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
                    return result/np.sqrt(2*self.g0/self.kappa_0), diff_array
                    break
            print("Converged in " + str(counter) + " iterations, relative error is " + str(diff))
            res = np.zeros(self.N_points,dtype=complex)
            res = Ak[index_1]
            for jj in ind_res:
                result[ind_modes,jj]= np.fft.fft(res[jj*N_m+ind_modes])
            return result/np.sqrt(2*self.g0/self.kappa_0), diff_array    
        
        def LinearStability(self,Seed_sol,dOm,plot_eigvals=True,get_eigvecs=True,order=0, IsSparse=False, NumOfEigVals=10,which='LM'):
            
            A=np.fft.ifft(Seed_sol, axis=0)
            A = A.T.reshape(A.size)*np.sqrt(2*self.g0/self.kappa_0)
            
                      
            N_m = self.N_points
            N_res = self.N_CROW
            
            index_1 = np.arange(0,N_m*N_res)
            index_2 = np.arange(N_m*N_res,2*N_m*N_res)
            ind_modes = np.arange(self.N_points)
            ind_res = np.arange(self.N_CROW)
            
            A_vec = np.zeros(2*N_m*N_res,dtype=complex)
            A_vec[index_1]=A
            A_vec[index_2]=np.conj(A)
            
            
            
             
            
            
            
            zeta_0 = dOm*2/self.kappa_0
            dphi = abs(self.phi[1]-self.phi[0])
            
            
            
            
            index_1 = np.arange(0,self.N_points)
            index_2 = np.arange(self.N_points,2*self.N_points)
            
            self.D = self.DispersionMatrix(order=order)
            self.M_lin = self.LinearMatrix(zeta_0)
            
            if IsSparse==True:
                Full_Matrix=(self.JacobianMatrix(zeta_0,A_vec))
                
                eig_vals,eig_vec = scp_eigs(Full_Matrix,k=NumOfEigVals,which=which)
            else:    
                Full_Matrix=(self.JacobianMatrix(zeta_0,A_vec))
                if get_eigvecs==True:
                    eig_vals,eig_vec = np.linalg.eig(Full_Matrix)
                if get_eigvecs==False:
                    eig_vals = np.linalg.eigvals(Full_Matrix)
            
            
            if plot_eigvals==True:
                plt.scatter(np.real(eig_vals),np.imag(eig_vals))
                plt.xlabel('Real part')
                plt.ylabel('Imaginary part')
            if get_eigvecs==True:
                eigen_vectors = np.zeros([self.N_points,self.N_CROW,2*self.N_points*self.N_CROW],dtype=complex)        
                for jj in range(2*self.N_points*self.N_CROW):
                    for ii in ind_res:
                        eigen_vectors[:,ii,jj]=(eig_vec[ii*N_m+ind_modes,jj]).T
                        eigen_vectors[:,ii,jj]=np.fft.fft(eigen_vectors[:,ii,jj])
                
            if get_eigvecs==True:    
                return eig_vals*self.kappa_0/2, eigen_vectors/np.sqrt(2*self.g0/self.kappa_0)
            if get_eigvecs==False:    
                return eig_vals*self.kappa_0/2
class Lattice(Resonator):  
    pass

def Plot_Map(map_data, detuning, colormap = 'cubehelix'):
    dOm = detuning[1]-detuning[0]
    dt=1
   
   
    Num_of_modes = map_data[0,:].size
    mu = np.arange(-Num_of_modes/2,Num_of_modes/2)
    def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero
    
        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to 
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }
    
        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)
    
        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False), 
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])
    
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)
    
            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))
    
        newcmap = mcolors.LinearSegmentedColormap(name, cdict)
        plt.register_cmap(cmap=newcmap)
    
        return newcmap


    def onclick(event):
        
        ix, iy = event.xdata, event.ydata
        x = int(np.floor((ix-detuning.min())/dOm))
        max_val = (abs(map_data[x,:])**2).max()
        plt.suptitle('Chosen detuning '+r'$\zeta_0$'+ '= %f'%ix, fontsize=20)
        ax.lines.pop(0)
        ax.plot([ix,ix], [-np.pi, np.pi ],'r')

        ax2 = plt.subplot2grid((5, 1), (2, 0))            
        ax2.plot(phi, abs(map_data[x,:])**2/max_val, 'r')
        ax2.set_ylabel('Intracavity power [a.u.]')
        ax2.set_xlim(-np.pi,np.pi)
        ax2.set_ylim(0,1)        
        ax3 = plt.subplot2grid((5, 1), (3, 0))
        ax3.plot(phi, np.angle(map_data[x,:])/(np.pi),'b')
#        if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
#            ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
        ax3.set_xlabel(r'$\varphi$')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_xlim(-np.pi,np.pi)
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
        ax3.grid(True)
        
        ax4 = plt.subplot2grid((5, 1), (4, 0))            
        ax4.plot(mu,10*np.log10(abs(np.fft.fftshift(np.fft.fft(map_data[x,:])))**2/(abs(np.fft.fft(map_data[x,:]))**2).max()),'-o', color='black',markersize=3)
        ax4.set_ylabel('Spectrum, dB')
        ax4.set_xlim(mu.min(),mu.max())
        #ax4.set_ylim(-100,3)   
        plt.show()
        f.canvas.draw()
        
    
    f = plt.figure()
    ax = plt.subplot2grid((5, 1), (0, 0), rowspan=2)
    plt.suptitle('Choose the detuning', fontsize=20)
    f.set_size_inches(10,8)
    phi = np.linspace(-np.pi,np.pi,map_data[0,:].size)
#    orig_cmap = plt.get_cmap('viridis')
#    colormap = shiftedColorMap(orig_cmap, start=0., midpoint=.5, stop=1., name='shrunk')
    pc = ax.pcolormesh(detuning, phi, abs(np.transpose(map_data))**2, cmap=colormap)
    ax.plot([0, 0], [-np.pi, np.pi], 'r')
    ax.set_xlabel('Detuning')
    ax.set_ylabel(r'$\varphi$')
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlim(detuning.min(),detuning.max())
    ix=0
    
    x = int(((ix-detuning.min())/dOm))
    if (x<0) or (x>detuning.size):
        x = 0
    max_val = (abs(map_data[x,:])**2).max()
    plt.suptitle('Chosen detuning '+r'$\zeta_0$'+ '= %f km'%ix, fontsize=20)
    ax.lines.pop(0)
    
    ax.plot([ix,ix], [-np.pi, np.pi ],'r')
    
    ax2 = plt.subplot2grid((5, 1), (2, 0))            
    ax2.plot(phi,abs(map_data[x,:])**2/max_val, 'r')
    ax2.set_ylabel('Intracavity power [a.u.]')
    ax2.set_xlim(-np.pi,np.pi)
    ax2.set_ylim(0,1)        
    ax3 = plt.subplot2grid((5, 1), (3, 0))
    ax3.plot(phi, np.angle(map_data[x,:])/(np.pi),'b')
#    if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
#        ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
    ax3.set_xlabel(r'$\varphi$')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_xlim(-np.pi,np.pi)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
    ax3.grid(True)
    ax4 = plt.subplot2grid((5, 1), (4, 0))            
    ax4.plot(mu,10*np.log10(abs(np.fft.fftshift(np.fft.fft(map_data[x,:])))**2/(abs(np.fft.fft(map_data[x,:]))**2).max()), '-o',color='black',markersize=3)
    ax4.set_ylabel('Spectrum, dB')
    ax4.set_xlim(mu.min(),mu.max())
    #ax4.set_ylim(-50,3)        
#    f.colorbar(pc)
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.93, wspace=None, hspace=0.4)
    f.canvas.mpl_connect('button_press_event', onclick)                


"""
here is a set of useful standard functions
"""
class FieldTheoryCROW:
    def __init__(self, resonator_parameters):
   #Physical parameters initialization
       
       
        self.J = resonator_parameters['Inter-resonator_coupling']
        self.N_CROW = resonator_parameters['N_res']
        self.N_theta = resonator_parameters['N_theta']
        self.D2 = resonator_parameters['D2']
        self.kappa_ex = resonator_parameters['kappa_ex']
        self.kappa_0 = resonator_parameters['kappa_0']
        self.kappa = self.kappa_0 + self.kappa_ex
        self.d2 = self.D2/self.kappa
        self.j = 2*self.J/self.kappa
        self.N_points = resonator_parameters['Number of modes']
        self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
        self.phi = np.linspace(0,2*np.pi,self.N_points)
        self.theta =np.linspace(0,2*np.pi,self.N_theta)
        self.delta_theta = 2*np.pi/self.N_CROW
    
    def seed_level (self, pump, detuning):
        f_norm = pump
        detuning_norm  = detuning
        res_seed = np.zeros_like(f_norm)
        res_seed = f_norm/(1-1j*detuning_norm)
        return res_seed
        #res_seed[0] = abs(np.min(stat_roots[ind_roots]))**.5
        
    def noise(self, a):
        return a*(np.random.uniform(-1,1,self.N_points*self.N_theta)+ 1j*np.random.uniform(-1,1,self.N_points*self.N_theta))

    def Propagate_SAMCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4,HardSeed=False):
        
        
        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        
        
        if HardSeed == False:
            seed = self.seed_level(Pump, detuning[0])
        else:
            seed = Seed.T.reshape(Seed.size)
        ### renormalization
        T_rn = (self.kappa/2)*T
        #f0 = np.fft.ifft(Pump,axis=0)*self.N_points
        f0=Pump
        print('f0='+str(f0.max()))
        print('xi [' + str(detuning[0]) + ',' +str(detuning[-1])+ '] (normalized on ' r'$kappa/2)$')
        noise_const = self.noise(eps) # set the noise level
        nn = len(detuning)
        
        t_st = float(T_rn)/len(detuning)
            
        sol = np.ndarray(shape=(len(detuning), self.N_points, self.N_theta), dtype='complex') # define an array to store the data
        
        ind_modes = np.arange(self.N_points)
        ind_res = np.arange(self.N_CROW)
        j = self.j
        kappa = self.kappa
        seed = seed.T.reshape(seed.size)
        for ii in range(self.N_theta):
            sol[0,ind_modes,ii] = seed[ii*self.N_points+ind_modes]
        
        f0 =(f0.T.reshape(f0.size))
        #%% crtypes defyning
        CROW_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_2D_lle_core.so')
        #else:
        #    sys.exit('Solver has not been found')
        
        CROW_core.PropagateSAM.restype = ctypes.c_void_p
        
        A = np.zeros([self.N_theta*self.N_points],dtype=complex)
        for ii in range(self.N_theta):    
        #    A[ii*self.N_points+ind_modes] = np.fft.ifft( seed[ii*self.N_points+ind_modes])*self.N_points
            A[ii*self.N_points+ind_modes] =  seed[ii*self.N_points+ind_modes]
        
    
        In_val_RE = np.array(np.real(A),dtype=ctypes.c_double)
        In_val_IM = np.array(np.imag(A),dtype=ctypes.c_double)
        In_phi = np.array(self.phi,dtype=ctypes.c_double)
        In_Nphi = ctypes.c_int(self.N_points)
        In_Ncrow = ctypes.c_int(self.N_CROW)
        In_theta = np.array(self.theta,dtype=ctypes.c_double)
        In_Ntheta = ctypes.c_int(self.N_theta)
        In_f_RE = np.array(np.real(f0 ),dtype=ctypes.c_double)
        In_f_IM = np.array(np.imag(f0 ),dtype=ctypes.c_double)
        In_atol = ctypes.c_double(abtol)
        In_rtol = ctypes.c_double(reltol)
        In_d2 = ctypes.c_double(self.d2)
        In_j = ctypes.c_double(self.j)
        In_kappa = ctypes.c_double(self.kappa)
        In_delta_theta = ctypes.c_double(self.delta_theta)

        In_det = np.array(detuning,dtype=ctypes.c_double)
        In_Ndet = ctypes.c_int(len(detuning))
        
        In_Tmax = ctypes.c_double(t_st)
        In_Nt = ctypes.c_int(int(t_st/dt)+1)
        In_dt = ctypes.c_double(dt)
        In_noise_amp = ctypes.c_double(eps)
        
        In_res_RE = np.zeros(len(detuning)*self.N_points*self.N_theta,dtype=ctypes.c_double)
        In_res_IM = np.zeros(len(detuning)*self.N_points*self.N_theta,dtype=ctypes.c_double)
        
        double_p=ctypes.POINTER(ctypes.c_double)
        In_val_RE_p = In_val_RE.ctypes.data_as(double_p)
        In_val_IM_p = In_val_IM.ctypes.data_as(double_p)
        In_phi_p = In_phi.ctypes.data_as(double_p)
        In_theta_p = In_theta.ctypes.data_as(double_p)
        In_det_p = In_det.ctypes.data_as(double_p)
        
        In_f_RE_p = In_f_RE.ctypes.data_as(double_p)
        In_f_IM_p = In_f_IM.ctypes.data_as(double_p)
        
        In_res_RE_p = In_res_RE.ctypes.data_as(double_p)
        In_res_IM_p = In_res_IM.ctypes.data_as(double_p)
        
        CROW_core.PropagateSAM(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_phi_p, In_theta_p, In_delta_theta, In_d2, In_j, In_Ndet, In_Nt, In_dt, In_atol, In_rtol, In_Nphi, In_Ntheta, In_noise_amp, In_res_RE_p, In_res_IM_p)
            
        
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            for jj in range(self.N_theta):
                sol[ii,ind_modes,jj] = (In_res_RE[ii*self.N_points*self.N_theta + jj*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points*self.N_theta+ jj*self.N_points+ind_modes])
            
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol
        elif out_param == 'fin_res':
            return sol[-1, :]
        else:
            print ('wrong parameter')



if __name__ == '__main__':
    print('PyCORe')
    
