#include "periodic_crow_core.hpp"

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
    
    std::complex<double>* noise_spectrum = new (std::nothrow) std::complex<double>[N];//contains white noise in spectal domain
    std::complex<double>* res = new (std::nothrow) std::complex<double>[N];//contains white noise in spectal domain
    fftw_complex noise_direct[N];
    fftw_plan p;
    
    p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(noise_spectrum), noise_direct, FFTW_BACKWARD, FFTW_ESTIMATE);
    double phase;
    double noise_amp;
    const std::complex<double> i(0, 1);
    std::default_random_engine generator;
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

void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double* kappa, const double kappa0, const double* J, const double *phi,  const double* d2, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Core on C++ is running";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi*Ncrow];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi*Ncrow];
    VecDoub res_buf(2*Nphi*Ncrow);


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

    Output out;
    rhs_crow crow(Nphi, Ncrow, detuning[0], f, d2,phi,std::abs(phi[1]-phi[0]),J, kappa, kappa0);

    
    std::cout<<"Step adaptative Dopri853 from NR3 is running\n";
    for (int i_det=0; i_det<Ndet; i_det++){
        crow.det = detuning[i_det]*2/kappa0; 
        noise=WhiteNoise(noise_amp,Nphi*Ncrow);
        Odeint<StepperDopr853<rhs_crow> > ode(res_buf,t0,t1,atol,rtol,dt,dtmin,out,crow);
        ode.integrate();
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
    std::cout<<"Step adaptative Dopri853 from NR3 is finished\n";
}

