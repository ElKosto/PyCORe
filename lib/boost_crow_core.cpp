#include "boost_crow_core.hpp"

void printProgress (double percentage)
{
    int val = (int) (percentage*100 );
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}
std::complex<double>* WhiteNoise(const double amp, const int N)
{
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::complex<double>* noise_spectrum = new (std::nothrow) std::complex<double>[N];//contains white noise in spectal domain
    std::complex<double>* res = new (std::nothrow) std::complex<double>[N];//contains white noise in spectal domain
    fftw_complex noise_direct[N];
    fftw_plan p;
    
    p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(noise_spectrum), noise_direct, FFTW_BACKWARD, FFTW_ESTIMATE);
    double phase;
    double noise_amp;
    const std::complex<double> i(0, 1); 
    std::default_random_engine generator(seed1);
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int j=0; j<N; j++){
       phase = distribution(generator) *2*M_PI-M_PI;
       noise_amp  = distribution(generator) *amp;
       noise_spectrum[j] = noise_amp *std::exp(i*phase)/sqrt(N);
    }   


    fftw_execute(p);
    for (int j=0; j<N; j++){
        res[j].real(noise_direct[j][0]);
        res[j].imag(noise_direct[j][1]);
    }   
    fftw_destroy_plan(p);
    delete [] noise_spectrum;
    return res;
}


void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double* kappa, const double kappa0, const double *delta, const double* J, const double *phi,  const double* Dint, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM)
{

    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi*Ncrow];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi*Ncrow];
    double a_x = 1., a_dxdt=1.;
    state_type res_buf(2*Nphi*Ncrow);



    noise=WhiteNoise(noise_amp,Nphi*Ncrow);
    for (int i_phi = 0; i_phi<Nphi*Ncrow; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
    }
    for (int i_crow = 0; i_crow<Ncrow; i_crow++){
        for (int i_phi=0; i_phi<Nphi; i_phi++ ){
            res_buf[i_crow*2*Nphi+i_phi] = res_RE[i_crow*Nphi+i_phi] + noise[i_crow*Nphi+i_phi].real();
            res_buf[i_crow*2*Nphi+i_phi+Nphi] = res_IM[i_crow*Nphi+i_phi] + noise[i_crow*Nphi+i_phi].imag();
            f[i_crow*2*Nphi+i_phi] = Re_F[i_crow*Nphi+i_phi];
            f[i_crow*2*Nphi+i_phi+Nphi] = Im_F[i_crow*Nphi+i_phi];
        }
    }
    rhs_pseudo_spectral_crow crow(Nphi, Ncrow, detuning[0], f, Dint,phi,std::abs(phi[1]-phi[0]),J, kappa, kappa0, delta);

    typedef boost::numeric::odeint::runge_kutta_fehlberg78< state_type > error_stepper_type;
    typedef boost::numeric::odeint::controlled_runge_kutta< error_stepper_type > controlled_stepper_type;
    controlled_stepper_type controlled_stepper(
        boost::numeric::odeint::default_error_checker< double , boost::numeric::odeint::range_algebra , boost::numeric::odeint::default_operations >( atol , rtol , a_x , a_dxdt ) );

    std::cout<<"Pseudo Spectral Step adaptative runge_kutta_fehlberg78 from boost::ode is running\n";
    for (int i_det=0; i_det<Ndet; i_det++){
        crow.det = detuning[i_det]*2/kappa0;
        noise=WhiteNoise(noise_amp,Nphi*Ncrow);
        boost::numeric::odeint::integrate_adaptive( controlled_stepper , crow , res_buf , t0 , t1, dt );
        //boost::numeric::odeint::integrate( crow , res_buf , t0 , t1, dt );
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                res_RE[i_det*Nphi* Ncrow+ i_crow*Nphi + i_phi] = res_buf[i_crow*2*Nphi+i_phi];
                res_IM[i_det*Nphi*Ncrow + i_crow*Nphi + i_phi] = res_buf[i_crow*2*Nphi+i_phi+Nphi];
                res_buf[i_crow*2*Nphi+i_phi] += noise[i_crow*Nphi+i_phi].real();
                res_buf[i_crow*2*Nphi+i_phi+Nphi] += noise[i_crow*Nphi+i_phi].imag();

            }
        }
        printProgress((i_det+1.)/Ndet);

    }
    delete [] noise;
    delete [] f;
    std::cout<<"Pseudo Spectral Step adaptative runge_kutta_fehlberg78 from boost::ode is finished\n";
    return 0;
}
