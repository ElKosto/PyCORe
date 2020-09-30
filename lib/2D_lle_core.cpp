#include "2D_lle_core.hpp"

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

void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double J, const double *phi, const double* theta, const double delta_theta, const double d2, const double j2,  const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, const int Ntheta, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Core on C++ is running";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi*Ntheta];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi*Ntheta];
    VecDoub res_buf(2*Nphi*Ntheta);


    noise=WhiteNoise(noise_amp,Nphi*Ntheta);
    for (int i_phi = 0; i_phi<Nphi*Ntheta; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
    }
    for (int i_theta = 0; i_theta<Ntheta; i_theta++){
        for (int i_phi=0; i_phi<Nphi; i_phi++ ){
            res_buf[i_theta*2*Nphi+i_phi] = res_RE[i_theta*Nphi+i_phi] + noise[i_theta*Nphi+i_phi].real();
            res_buf[i_theta*2*Nphi+i_phi+Nphi] = res_IM[i_theta*Nphi+i_phi] + noise[i_theta*Nphi+i_phi].imag();
            f[i_theta*2*Nphi+i_phi] = Re_F[i_theta*Nphi+i_phi];
            f[i_theta*2*Nphi+i_phi+Nphi] = Im_F[i_theta*Nphi+i_phi];
        }
    }

    Output out;
    rhs_crow crow(Nphi, Ntheta, detuning[0], f, d2,phi,std::abs(phi[1]-phi[0]), theta, std::abs(theta[1]-theta[0]), J, delta_theta);

    
    std::cout<<"Step adaptative Dopri853 from NR3 is running\n";
    for (int i_det=0; i_det<Ndet; i_det++){
        crow.det = detuning[i_det]; 
        noise=WhiteNoise(noise_amp,Nphi*Ntheta);
        Odeint<StepperDopr853<rhs_crow> > ode(res_buf,t0,t1,atol,rtol,dt,dtmin,out,crow);
        ode.integrate();
        for (int i_theta= 0; i_theta<Ntheta; i_theta++){
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                res_RE[i_det*Nphi* Ntheta+ i_theta*Nphi + i_phi] = res_buf[i_theta*2*Nphi+i_phi];
                res_IM[i_det*Nphi*Ntheta + i_theta*Nphi + i_phi] = res_buf[i_theta*2*Nphi+i_phi+Nphi];
                res_buf[i_theta*2*Nphi+i_phi] += noise[i_theta*Nphi+i_phi].real();
                res_buf[i_theta*2*Nphi+i_phi+Nphi] += noise[i_theta*Nphi+i_phi].imag();

            }
        }
        printProgress((i_det+1.)/Ndet);

    }
    delete [] noise;
    delete [] f;
    std::cout<<"Step adaptative Dopri853 from NR3 is finished\n";
}

