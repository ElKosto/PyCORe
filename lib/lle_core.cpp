#include "lle_core.hpp"

void printProgress (double percentage)
{
    int val = (int) (percentage*100 );
    int lpad = (int) (percentage * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
    fflush (stdout);
}
std::complex<double>* WhiteNoise(const double amp, const int Nphi)
{
    
    std::complex<double>* noise_spectrum = new (std::nothrow) std::complex<double>[Nphi];//contains white noise in spectal domain
    std::complex<double>* res = new (std::nothrow) std::complex<double>[Nphi];//contains white noise in spectal domain
    fftw_complex noise_direct[Nphi];
    fftw_plan p;
    
    p = fftw_plan_dft_1d(Nphi, reinterpret_cast<fftw_complex*>(noise_spectrum), noise_direct, FFTW_BACKWARD, FFTW_ESTIMATE);
    double phase;
    double noise_amp;
    const std::complex<double> i(0, 1);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    for (int j=0; j<Nphi; j++){
       phase = distribution(generator) *2*M_PI-M_PI;
       noise_amp  = distribution(generator) *amp;
       noise_spectrum[j] = noise_amp *std::exp(i*phase)/sqrt(Nphi);
    }


    fftw_execute(p);
    for (int j=0; j<Nphi; j++){
        res[j].real(noise_direct[j][0]);
        res[j].imag(noise_direct[j][1]);
    }
    fftw_destroy_plan(p);
    delete [] noise_spectrum;
    return res;
}
void SaveData( std::complex<double> **A, const int Ndet, const int Nphi)
{

    std::ofstream outFile;
    outFile.open("Field.bin", std::ios::binary);
    for (int i =0; i<Ndet; i++){
        for (int j=0; j<Nphi; j++){
            outFile.write(reinterpret_cast<const char*>(&A[i][j]),sizeof(std::complex<double>));
        }
    }
    outFile.close();
}
void* PropagateSS(double* In_val_RE, double* In_val_IM, const double f,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Split Step is running\n";
    using namespace std::complex_literals;
    std::complex<double> i = 1i;
    int check;
    
    /*std::complex<double> **res = new (std::nothrow) std::complex<double>*[Ndet];
    for (int i=0; i<Ndet; i++){
        res[i] = new (std::nothrow) std::complex<double>[Nphi];
    }*/
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    
    //res[0] = InitialValue(f, detuning[0], Nphi);
    
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
    }
    noise=WhiteNoise(noise_amp,Nphi);
    
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    
    std::complex<double> buf;
    fftw_complex buf_direct[Nphi], buf_spectrum[Nphi];
    plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_PATIENT);
    plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_PATIENT);
    
    for (int i_phi=0; i_phi<Nphi; i_phi++){
          buf_direct[i_phi][0] = res_RE[i_phi] + noise[i_phi].real();
          buf_direct[i_phi][1] = res_IM[i_phi] + noise[i_phi].imag();
    }
    fftw_execute(plan_direct_2_spectrum);

    for (int i_det=0; i_det<Ndet; i_det++){
        noise=WhiteNoise(noise_amp,Nphi);
        for (int i_t=0; i_t<Nt; i_t++){
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf.real( buf_direct[i_phi][0] );
                buf.imag( buf_direct[i_phi][1]);
                buf+=(noise[i_phi]);
                buf*= std::exp(dt *(i*buf*std::conj(buf)  +i*J*(std::cos(phi[i_phi])+0.*std::sin(2*phi[i_phi]))  ) );
                buf_direct[i_phi][0] = buf.real();
                buf_direct[i_phi][1] = buf.imag();
            }
            fftw_execute(plan_direct_2_spectrum);//First step terminated
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf.real( buf_spectrum[i_phi][0]);
                buf.imag( buf_spectrum[i_phi][1]);
                buf *= std::exp(dt * (-1. - i*detuning[i_det] - i*Dint[i_phi] + f*Nphi/buf*((i_phi==0)? 1.0 : 0.0) )  ); 
                buf_spectrum[i_phi][0] = buf.real()/Nphi;
                buf_spectrum[i_phi][1] = buf.imag()/Nphi;
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
    delete [] noise;
    fftw_destroy_plan(plan_direct_2_spectrum);
    fftw_destroy_plan(plan_spectrum_2_direct);
    std::cout<<"Split step is finished\n";
}

