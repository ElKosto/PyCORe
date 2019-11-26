import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import complex_ode
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg


class Resonator:
    def __init__(self, res_param):
        self.FSR = res_param['FSR']
        self.n0 = res_param['n0']
        self.mu = np.fft.ifftshift(np.arange(-self.N_points/2,self.N_points/2))
        self.Seed =  SeedField
        self.TimeStep = dT
        self.Dint = Dint
        self.L = L
        self.Tr = Tr
        self.gamma = gamma
        self.kappa_0 = kappa_0
        self.kappa_ex = kappa_ex
        self.kappa = kappa_ex + kappa_0
        self.Traman = Traman
        # spectral noise 

    def noise(self, a):
        return a*np.exp(1j*np.random.uniform(-1,1,self.N_points)*np.pi)

    #   Propagate Using the Step Adaptive  Method
    def Propagate_SAM(self, simulation_param):
        T = simulation_parameters['slow_time']
        abtol = simulation_parameters['absolute_tolerance']
        reltol = simulation_parameters['relative_tolerance']
        out_param = simulation_parameters['output']
        nmax = simulation_parameters['max_internal_steps']
        dOm = simulation_parameters['detuning_array']
        eps = simulation_parameters['noise_level']
          
        noise_const = self.noise(eps) # set the noise level
        nn = len(dOm)
        
        def disp(field_in,Dint_in):
        # computes the dispersion term in Fourier space
            field_fft = np.fft.fft(field_in)
            out_field = np.fft.ifft(Dint_in*field_fft)     
            return out_field
        ### define the rhs function
        def LLE_1d(Z, A):
            # for nomalized
            if np.size(self.Dint)==1 and self.Dint == 1:
                A = -noise_const
                norm = 2*np.pi/len(A)
                A_dir = np.fft.ifft(A)/norm
                B_dir = np.fft.ifft(B)/norm
                dAdz =  -.5*(kappa_0[0]+kappa_ex[0])*A + 1j*(Dint[:,0] + dOm_curr)*A - 1j*gamma*np.fft.fft(A_dir*np.abs(A_dir)**2)*norm - 1j*g*np.exp(-1j*mu*np.pi)*B + np.sqrt(kappa_ex[0])*self.pump
            elif np.size(self.Dint)==1 and self.Dint == -1:
                 dAdt2 = deriv_2(self.TimeStep, A)
                 dAdT =  -1j*dAdt2/2 + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump
            else:  
                # without raman
                Disp_int = disp(A,self.Dint)
                if self.Traman==0:
                    dAdT =  -1j*Disp_int + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump
                else:
                    # with raman
                    dAAdt = deriv_1(self.TimeStep,abs(A)**2)
                    dAdT =  -1j*Disp_int + 1j*self.gamma*self.L/self.Tr*np.abs(A)**2*A - (self.kappa/2+1j*dOm_curr)*A -1j*self.gamma*self.Traman*dAAdt*A + np.sqrt(self.kappa/2/self.Tr)*self.Pump
            return dAdT
        
        t_st = float(T)/len(detuning)
        r = complex_ode(LLE_1d).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        r.set_initial_value(self.Seed, 0)# seed the cavity
        sol = np.ndarray(shape=(len(detuning), len(self.Seed)), dtype='complex') # define an array to store the data
        sol[0,:] = self.Seed
        printProgressBar(0, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for it in range(1,len(detuning)):
            printProgressBar(it + 1, n, prefix = 'Progress:', suffix = 'Complete', length = 50)
            dOm_curr = detuning[it] # detuning value
            sol[it] = r.integrate(r.t+t_st)
            
        if out_param == 'map':
            return sol
        elif out_param == 'fin_res':
            return sol[-1, :] 
        else:
            print ('wrong parameter')

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
        
        r = complex_ode(LLE_1d).set_integrator('dop853', atol=abtol, rtol=reltol,nsteps=nmax)# set the solver
        r.set_initial_value(self.Seed, 0)# seed the cavity
        
        
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
        line, = plt.plot(abs(self.Seed)**2)
        plt.ylim(0,1.1)
        plt.ylabel('$|\Psi|^2$')
        
        ax3 = plt.subplot(224)
        line2, = plt.semilogy(self.mu, np.abs(np.fft.fft(self.Seed))**2)
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
            
                
    ### function to seed the soliton
    def seed_soliton_norm(fast_t, f_pump, detun):
        stat_roots = np.roots([1, -2*detun, (detun**2+1), -f_pump**2])
        ind_roots = [np.imag(ii)==0 for ii in stat_roots]
        B = np.sqrt(2*detun)
        return np.min(np.abs(stat_roots[ind_roots]))**.5 + B*np.exp(1j*np.arccos(2*B/np.pi/f_pump)*2)*np.cosh(B*fast_t)**-1


class CROW(Resonator):

    def __init__(self, dNu, Pavg, Offset=0, NofP=1e4, dT=0.05):
        self.SpWidth = dNu
        self.AvgPower = Pavg
        self.Span = NofP*dT
        self.TimeStep = dT
        Rf = np.fft.ifft(np.exp(-1*(np.fft.fftfreq(int(NofP),d=dT))**2/4/((dNu/2)**2/2/np.log(2)))*np.exp(1j*np.random.uniform(-1,1,int(NofP))*np.pi))
        # Above 4 is for the sqrt of intensity 
        if Offset == 0:
            A = np.abs(Rf)**2
            A = Pavg*A/np.mean(A)
            Ph = np.angle(Rf)
            self.Sig = np.sqrt(A)*np.exp(1j*Ph)
        else:
            A = Rf*Pavg**.5+Offset
            self.Sig = A
            
class Lattice(Resonator):  
    pass
            
            
def Plot_Map(map_data,dt=1,dz=1,colormap = 'cubehelix',z0=0):
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
        x = int(np.floor(ix/dz))
        plt.suptitle('Chosen distance z = %f km'%ix, fontsize=20)
        ax.lines.pop(0)
        ax.plot([ix,ix], [0, dt*np.size(map_data,1)],'r')

        ax2 = plt.subplot2grid((4, 1), (2, 0))            
        ax2.plot(np.arange(0,dt*np.size(map_data,1),dt), abs(map_data[x,:])**2, 'r')
        ax2.set_ylabel('Power (W)')
        ax2.set_xlim(0, dt*np.size(map_data,1))        
        ax3 = plt.subplot2grid((4, 1), (3, 0))
        ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.angle(map_data[x,:])/(np.pi),'b')
        if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
            ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
        ax3.set_xlabel('Time (ps)')
        ax3.set_ylabel('Phase (rad)')
        ax3.set_xlim(0, dt*np.size(map_data,1))
        ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
        ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
        ax3.grid(True)
        plt.show()
        f.canvas.draw()
        
    f = plt.figure()
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
    plt.suptitle('Choose the coordinate', fontsize=20)
    f.set_size_inches(10,8)
    Z,T = np.meshgrid( np.arange(0,dz*np.size(map_data,0),dz), np.arange(0, dt*np.size(map_data,1),dt))
#    orig_cmap = plt.get_cmap('viridis')
#    colormap = shiftedColorMap(orig_cmap, start=0., midpoint=.5, stop=1., name='shrunk')
    pc = ax.pcolormesh(Z, T, abs(np.transpose(map_data))**2, cmap=colormap)
    ax.plot([0, 0], [0, dt*np.size(map_data,1)-dt], 'r')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Time (ps)')
    ax.set_ylim(0, dt*np.size(map_data,1))
    ax.set_xlim(0, dz*np.size(map_data,0)-5*dz)
    ix=z0
    x = int(np.floor(ix/dz))
    plt.suptitle('Chosen distance z = %f km'%ix, fontsize=20)
    ax.lines.pop(0)
    
    ax.plot([ix,ix], [0, dt*np.size(map_data,1)],'r')

    ax2 = plt.subplot2grid((4, 1), (2, 0))            
    ax2.plot(np.arange(0,dt*np.size(map_data,1),dt), abs(map_data[x,:])**2, 'r')
    ax2.set_ylabel('Power (W)')
    ax2.set_xlim(0, dt*np.size(map_data,1))        
    ax3 = plt.subplot2grid((4, 1), (3, 0))
    ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.angle(map_data[x,:])/(np.pi),'b')
    if max( np.unwrap(np.angle(map_data[x,:]))/(np.pi)) - min( np.unwrap(np.angle(map_data[x,:]))/(np.pi))<10:
        ax3.plot(np.arange(0,dt*np.size(map_data,1),dt), np.unwrap(np.angle(map_data[x,:]))/(np.pi),'g')
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_xlim(0, dt*np.size(map_data,1))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(base=1.0))
    ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g $\pi$'))
    ax3.grid(True)
#    f.colorbar(pc)
    plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.93, wspace=None, hspace=0.4)
    f.canvas.mpl_connect('button_press_event', onclick)
"""
here is a set of useful standard functions
"""
