#include "lle_core.h"

void printProgress (double percentage)
{
    int val = (int) (percentage*100 );
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}
complex* WhiteNoise(const double amp, const int Nphi)
{
    fftw_complex noise_direct[Nphi];
    fftw_complex noise_spectrum[Nphi];
    complex *res;
    res = malloc(Nphi*sizeof(complex));
    fftw_plan p;
    
    p = fftw_plan_dft_1d(Nphi, noise_spectrum, noise_direct, FFTW_BACKWARD, FFTW_ESTIMATE);
    double phase;
    double noise_amp;
    for (int j=0; j<Nphi; j++){
       phase =  (double)rand()/((double)(RAND_MAX/(2*M_PI))) - M_PI;
       noise_amp  = (double)rand()/((double)(RAND_MAX/(amp)));
       noise_spectrum[j][0] = noise_amp *cos(phase)/sqrt(Nphi);
       noise_spectrum[j][1] = noise_amp *sin(phase)/sqrt(Nphi);
    }

    fftw_execute(p);
    
    for (int j=0; j<Nphi; j++){
        res[j]= noise_direct[j][0]+I*noise_direct[j][1];
    }
    fftw_destroy_plan(p);
    return res;
}
void* PropagateSS(double* In_val_RE, double* In_val_IM, const double f,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
{
    
    printf("Split Step is running\n");
    complex i = CMPLX(0,1);
    complex buf;
    complex* noise;
    noise = malloc(Nphi*sizeof(complex));
    
    
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
    }
    noise=WhiteNoise(noise_amp,Nphi);
    
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    
    fftw_complex buf_direct[Nphi], buf_spectrum[Nphi];
    plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_PATIENT);
    plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_PATIENT);
    
    for (int i_phi=0; i_phi<Nphi; i_phi++){
          buf_direct[i_phi][0] = res_RE[i_phi] + creal(noise[i_phi]);
          buf_direct[i_phi][1] = res_IM[i_phi] + cimag(noise[i_phi]);
    }
    fftw_execute(plan_direct_2_spectrum);

    for (int i_det=0; i_det<Ndet; i_det++){
        noise=WhiteNoise(noise_amp,Nphi);
        for (int i_t=0; i_t<Nt; i_t++){
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf = buf_direct[i_phi][0] + I * buf_direct[i_phi][1];
                //buf.real( buf_direct[i_phi][0] );
                //buf.imag( buf_direct[i_phi][1]);
                buf+=(noise[i_phi]);
                buf*= exp(dt *(i*buf*conj(buf)  +I*J*(cos(phi[i_phi])+0.*sin(2*phi[i_phi]))  ) );
                buf_direct[i_phi][0] = creal(buf);
                buf_direct[i_phi][1] = cimag(buf);
            }
            fftw_execute(plan_direct_2_spectrum);//First step terminated
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf = buf_spectrum[i_phi][0] + I * buf_spectrum[i_phi][1];
                //buf.real( buf_spectrum[i_phi][0]);
                //buf.imag( buf_spectrum[i_phi][1]);
                buf *= exp(dt * (-1. - I*detuning[i_det] - I*Dint[i_phi] + f*Nphi/buf*((i_phi==0)? 1.0 : 0.0) )  ); 
                buf_spectrum[i_phi][0] = creal(buf)/Nphi;
                buf_spectrum[i_phi][1] = cimag(buf)/Nphi;
            }
            fftw_execute(plan_spectrum_2_direct);
            //Second step terminated
        }
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = buf_spectrum[i_phi][0];
            res_IM[i_det*Nphi+i_phi] = buf_spectrum[i_phi][1];
        }
        //std::cout<<(i_det+1.)/Ndet*100.<<"% is done\n";
        printProgress((i_det+1.)/Ndet);
    }
    
    
    //SaveData(res, Ndet, Nphi);
    free(noise);
    noise = NULL;
    fftw_destroy_plan(plan_direct_2_spectrum);
    fftw_destroy_plan(plan_spectrum_2_direct);
    printf("Split step is finished\n");
}

