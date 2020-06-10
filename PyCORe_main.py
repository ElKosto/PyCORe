import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode,solve_ivp, ode
from scipy.sparse.linalg import expm
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.constants import pi, c, hbar
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
from scipy.optimize import curve_fit
import time
import os
from scipy.sparse import block_diag,identity,diags, eye, csc_matrix
import ctypes



class Resonator:
    def __init__(self, resonator_parameters):
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
        #Auxiliary physical parameters
        self.Tr = 1/self.FSR #round trip time
        self.Aeff = self.width*self.height 
        self.Leff = c/self.n0*self.Tr 
        self.Veff = self.Aeff*self.Leff 
        self.g0 = hbar*self.w0**2*c*self.n2/self.n0**2/self.Veff
        self.gamma = self.n2*self.w0/c/self.Aeff
        self.kappa = self.kappa_0 + self.kappa_ex
        self.N_points = len(self.Dint)
        mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
        self.phi = np.linspace(0,2*np.pi,self.N_points)
        def func(x, a, b, c, d):
            return a + x*b + c*x**2/2 + d*x**3/6
        popt, pcov = curve_fit(func, mu, self.Dint)
        self.D2 = popt[2]
        self.D3 = popt[3]
        
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
        J =  simulation_parameters['electro-optical coupling']
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa,Normalized_Units)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa,Normalized_Units)
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
       
    def Propagate_SplitStep(self, simulation_parameters, Pump, Seed=[0], dt=5e-4, Normalized_Units=False):
        start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  simulation_parameters['electro-optical coupling']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        if Normalized_Units == False:
            pump = Pump*np.sqrt(1./(hbar*self.w0))
            if Seed[0] == 0:
                seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa,Normalized_Units)
            else:
                seed = Seed*np.sqrt(2*self.g0/self.kappa,Normalized_Units)
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
        f0 = np.fft.ifft(f0)*self.N_points
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
                #buf_dir = np.fft.ifft(buf)*len(buf)## in the direct space
                # First step
                #buf =buf + dt*(1j/len(buf)*np.fft.fft(buf_dir*np.abs(buf_dir)**2) + f0)
                buf = np.fft.fft(np.exp(dt*(1j*np.abs(buf)**2+1j*J*(np.cos(self.phi) + 0.*np.sin(2*self.phi)) + f0/buf))*buf)
                #second step
                #buf = np.exp(-dt *(1+1j*(self.Dint + dOm_curr)*2/self.kappa )) * buf
                buf = np.fft.ifft(np.exp(-dt *(1+1j*(self.Dint + dOm_curr)*2/self.kappa )) *buf)
                
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
            
    def Propagate_SplitStepCLIB(self, simulation_parameters, Pump, Seed=[0], dt=5e-4):
        #start_time = time.time()
        T = simulation_parameters['slow_time']
        out_param = simulation_parameters['output']
        detuning = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
        J =  simulation_parameters['electro-optical coupling']
        #dt = simulation_parameters['time_step']#in photon lifetimes
        
        pump = Pump*np.sqrt(1./(hbar*self.w0))
        
        if Seed[0] == 0:
            seed = self.seed_level(Pump, detuning[0])*np.sqrt(2*self.g0/self.kappa)
        else:
            seed = Seed*np.sqrt(2*self.g0/self.kappa)
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
        LLE_core = ctypes.CDLL(os.path.abspath(__file__)[:-15]+'/lib/lib_lle_core.so')
        LLE_core.PropagateSS.restype = ctypes.c_void_p
        #%% defining the ctypes variables
        
        A = np.fft.ifft(seed)*len(seed)
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
        LLE_core.PropagateSS(In_val_RE_p, In_val_IM_p, In_f_RE_p, In_f_IM_p, In_det_p, In_J, In_phi_p, In_Dint_p, In_Ndet, In_Nt, In_dt, In_Nphi, In_noise_amp, In_res_RE_p, In_res_IM_p)
        
        ind_modes = np.arange(self.N_points)
        for ii in range(0,len(detuning)):
            sol[ii,ind_modes] = In_res_RE[ii*self.N_points+ind_modes] + 1j*In_res_IM[ii*self.N_points+ind_modes]
        #sol = np.reshape(In_res_RE,[len(detuning),self.N_points]) + 1j*np.reshape(In_res_IM,[len(detuning),self.N_points])
                    
        if out_param == 'map':
            return sol/np.sqrt(2*self.g0/self.kappa)
        elif out_param == 'fin_res':
            return sol[-1, :]/np.sqrt(2*self.g0/self.kappa)
        else:
            print ('wrong parameter')
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
class CROW(Resonator):#all idenical resonators
        def __init__(self, resonator_parameters):
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
            self.N_CROW = len(self.Dint[0,:])
            self.D2 = np.zeros(self.N_CROW)
            self.D3 = np.zeros(self.N_CROW)
            self.kappa_ex = resonator_parameters['kappa_ex']# V
            self.kappa = self.kappa_0 + self.kappa_ex
            self.N_points = len(self.Dint[:,0])
            self.mu = np.fft.fftshift(np.arange(-self.N_points/2, self.N_points/2))
            self.phi = np.linspace(0,2*np.pi,self.N_points)
            def func(x, a, b, c, d):
                    return a + x*b + c*x**2/2 + d*x**3/6
            for ii in range(0,self.N_CROW):
                self.Dint[:,ii] = np.fft.ifftshift(self.Dint[:,ii])
                
                popt, pcov = curve_fit(func, self.mu, self.Dint[:,ii])
                self.D2[ii] = popt[2]
                self.D3[ii] = popt[3]
            
            ind_phase_modes = np.arange(0,(self.N_CROW-1)*self.N_points)
            ind_phase_modes = ind_phase_modes%self.N_points
            M_lin = diags(-(self.kappa.T.reshape(self.kappa.size)/self.kappa_0+1j*self.Dint.T.reshape(self.Dint.size)*2/self.kappa_0),0) + 1j*diags(self.J.T.reshape(self.J.size)*2/self.kappa_0 *np.exp(-1j*ind_phase_modes*np.pi),self.N_points) + 1j*diags(self.J.T.reshape(self.J.size)*2/self.kappa_0 *np.exp(1j*ind_phase_modes*np.pi),-self.N_points)
            
            self.M_lin = M_lin
            #self.M_lin = M_lin.todense()

            
           
            
        def seed_level (self, pump, detuning):
            f_norm = pump*np.sqrt(1./(hbar*self.w0))*np.sqrt(8*self.g0*self.kappa_ex/self.kappa_0**3)#we pump the first ring
            detuning_norm  = detuning*2/self.kappa_0
            
            
            LinearM = np.eye(self.N_points*self.N_CROW,dtype = complex)
            ind_modes = np.arange(self.N_points)
            for ii in range(0,self.N_CROW-1):
                LinearM[ind_modes+ii*self.N_points,ind_modes+(ii+1)*self.N_points] = 1j*self.J.T.reshape(self.J.size)[ii*self.N_points +ind_modes]*2/self.kappa_0
            LinearM += LinearM.T
            indM = np.arange(self.N_points*self.N_CROW)
            LinearM[indM,indM] = -(self.kappa.T.reshape(self.kappa.size)[indM]/self.kappa_0 + 1j*detuning_norm)
            
           
            
            res_seed = np.zeros_like(f_norm.reshape(f_norm.size))
            res_seed = np.linalg.solve(LinearM,f_norm.T.reshape(f_norm.size))
            res_seed*= 1/np.sqrt(2*self.g0/self.kappa_0)
            #res_seed.reshape((self.N_points,self.N_CROW))
            
            return res_seed
        def noise(self, a):
#        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)
            return a*(np.random.uniform(-1,1,self.N_points*self.N_CROW)+ 1j*np.random.uniform(-1,1,self.N_points*self.N_CROW))
        
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
            ### renarmalization
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
                    buf_vec = expm(dt*(self.M_lin -1j*dOm_curr*2/self.kappa_0* eye(self.N_points*self.N_CROW) )).dot(buf.T.reshape(buf.size))
                  
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
                dAdT[0*self.N_points+ind_modes] += 1j*self.J[:,0]*2/self.kappa_0 *np.exp(-1j*crow.mu*np.pi)*A[1*self.N_points+ind_modes]+1j*np.fft.fft(A_dir[0*self.N_points+ind_modes]*np.abs(A_dir[0*self.N_points+ind_modes])**2)/self.N_points
                dAdT[(self.N_CROW-1)*self.N_points+ind_modes] += 1j*self.J[:,self.N_CROW-2]*2/self.kappa_0 *np.exp(1j*crow.mu*np.pi)*A[((self.N_CROW-2))*self.N_points+ind_modes]+1j*np.fft.fft(A_dir[(self.N_CROW-1)*self.N_points+ind_modes]*np.abs(A_dir[(self.N_CROW-1)*self.N_points+ind_modes])**2)/self.N_points
                for ii in range(1,self.N_CROW-1):
                    dAdT[ii*self.N_points+ind_modes]+= 1j*self.J[:,ii]*2/self.kappa_0 *np.exp(-1j*crow.mu*np.pi)*A[(ii+1)*self.N_points+ind_modes] + 1j*self.J[:,ii-1]*2/self.kappa_0 *np.exp(1j*crow.mu*np.pi)*A[(ii-1)*self.N_points+ind_modes] +  1j*np.fft.fft(A_dir[ii*self.N_points+ind_modes]*np.abs(A_dir[ii*self.N_points+ind_modes])**2)/self.N_points
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
if __name__ == '__main__':
    print('PyCORe')
    