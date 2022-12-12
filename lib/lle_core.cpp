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
    unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
    std::complex<double>* noise_spectrum = new (std::nothrow) std::complex<double>[Nphi];//contains white noise in spectal domain
    std::complex<double>* res = new (std::nothrow) std::complex<double>[Nphi];//contains white noise in spectal domain
    fftw_complex noise_direct[Nphi];
    fftw_plan p;
    
    p = fftw_plan_dft_1d(Nphi, reinterpret_cast<fftw_complex*>(noise_spectrum), noise_direct, FFTW_BACKWARD, FFTW_ESTIMATE);
    double phase;
    double noise_amp;
    const std::complex<double> i(0, 1);
    std::default_random_engine generator(seed1);
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

void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Step adaptative Dopri853 from NR3 is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi];
    VecDoub res_buf(2*Nphi);


    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
        f[i_phi] = Re_F[i_phi];
        f[i_phi+Nphi] = Im_F[i_phi];
    }
    std::cout<<"In val_RE = " << In_val_RE[0]<<std::endl;

    Output out;
    rhs_lle lle(Nphi, Dint, detuning[0],f,Dint[1],phi,std::abs(phi[1]-phi[0]),J);
    
    for (int i_det=0; i_det<Ndet; i_det++){
        lle.det = detuning[i_det];
        noise=WhiteNoise(noise_amp,Nphi);
        Odeint<StepperDopr853<rhs_lle> > ode(res_buf,t0,t1,atol,rtol,dt,dtmin,out,lle);
        ode.integrate();
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = res_buf[i_phi];
            res_IM[i_det*Nphi+i_phi] = res_buf[i_phi+Nphi];
            res_buf[i_phi] += noise[i_phi].real();
            res_buf[i_phi+Nphi] += noise[i_phi].imag();

        }
        printProgress((i_det+1.)/Ndet);

    }
    delete [] noise;
    delete [] f;
//    delete [] res_buf;
    std::cout<<"Step adaptative Dopri853 from NR3 is finished\n";
}

void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
    
{
    
    std::cout<<"Pseudo Spectral Step adaptative Dopri853 from NR3 is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi];
    VecDoub res_buf(2*Nphi);


    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
        f[i_phi] = Re_F[i_phi];
        f[i_phi+Nphi] = Im_F[i_phi];
    }
    std::cout<<"In val_RE = " << In_val_RE[0]<<std::endl;

    Output out;
    rhs_pseudo_spectral_lle lle(Nphi, Dint, detuning[0],f,phi,std::abs(phi[1]-phi[0]),J);
    
    for (int i_det=0; i_det<Ndet; i_det++){
        lle.det = detuning[i_det];
        noise=WhiteNoise(noise_amp,Nphi);
        Odeint<StepperDopr853<rhs_pseudo_spectral_lle> > ode(res_buf,t0,t1,atol,rtol,dt,dtmin,out,lle);
        ode.integrate();
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = res_buf[i_phi];
            res_IM[i_det*Nphi+i_phi] = res_buf[i_phi+Nphi];
            res_buf[i_phi] += noise[i_phi].real();
            res_buf[i_phi+Nphi] += noise[i_phi].imag();

        }
        printProgress((i_det+1.)/Ndet);

    }
    delete [] noise;
    delete [] f;
//    delete [] res_buf;
    std::cout<<"Pseudo Spectral Step adaptative Dopri853 from NR3 is finished\n";
}
void* PropagateThermalSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double J, const double t_th, const double kappa, const double n2, const double n2t, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Step adaptative Dopri853 from NR3 with thermal effects is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi];
    VecDoub res_buf(2*Nphi+1);
    double power=0.;

    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        power+= res_RE[i_phi]*res_RE[i_phi] + res_IM[i_phi]*res_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
        f[i_phi] = Re_F[i_phi];
        f[i_phi+Nphi] = Im_F[i_phi];
    }
    res_buf[2*Nphi] = power;

    Output out;
    rhs_lle_thermal lle(Nphi, Dint, detuning[0],f,Dint[1],phi,std::abs(phi[1]-phi[0]),J, t_th, kappa, n2, n2t);
    
    for (int i_det=0; i_det<Ndet; i_det++){
        lle.det = detuning[i_det];
        noise=WhiteNoise(noise_amp,Nphi);
        Odeint<StepperDopr853<rhs_lle_thermal> > ode(res_buf,t0,t1,atol,rtol,dt,dtmin,out,lle);
        ode.integrate();
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = res_buf[i_phi];
            res_IM[i_det*Nphi+i_phi] = res_buf[i_phi+Nphi];
            res_buf[i_phi] += noise[i_phi].real();
            res_buf[i_phi+Nphi] += noise[i_phi].imag();

        }
        printProgress((i_det+1.)/Ndet);

    }
    delete [] noise;
    delete [] f;
//    delete [] res_buf;
    std::cout<<"Step adaptative Dopri853 from NR3 is finished\n";
}
void* PropagateSS(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Split Step is running\n";

    std::complex<double> i (0.,1.);
    std::complex<double> f;
    
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
                //buf*= std::exp(dt *(i*buf*std::conj(buf)  +i*J*(std::cos(phi[i_phi])+0.*std::sin(2*phi[i_phi]))  ) );
                buf*= std::exp(dt *(i*buf*std::conj(buf)    ) );
                buf_direct[i_phi][0] = buf.real();
                buf_direct[i_phi][1] = buf.imag();
            }
            fftw_execute(plan_direct_2_spectrum);//First step terminated
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf.real( buf_spectrum[i_phi][0]);
                buf.imag( buf_spectrum[i_phi][1]);
                f.real(Re_F[i_phi]*Nphi);
                f.imag(Im_F[i_phi]*Nphi);
                //buf *= std::exp(dt * (-1. - i*detuning[i_det] - i*Dint[i_phi] + f/buf )  ); 
                buf = std::exp(dt * (-1. - i*detuning[i_det] - i*Dint[i_phi] )  )*buf + f/(-1. - i*detuning[i_det] - i*Dint[i_phi])*(std::exp(dt * (-1. - i*detuning[i_det] - i*Dint[i_phi] ) ) - 1.); 
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

void* Propagate_SiL_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, const double *detuning, const double kappa, const double kappa_laser ,const double kappa_sc, const double kappa_inj, const double coupling_phase, const double g0, const double alpha, const double gamma, const double V, const double a, const double e, const double N0, const double eta, const double I_laser, const double zeta, const double* Dint, const int Ndet, const double Tmax, const double T_step, const int Nt, const double dt, const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
{
    
    std::cout<<"Step adaptative Dopri853 from NR3 for Self-Injection locked resonator is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    const double t0=0., t1=T_step, dtmin=0.;
    double t_step_ini=0., t_step_end=0.;
    double tuning_speed;
    tuning_speed = (detuning[Ndet-1] - detuning[0])/Tmax;
    std::cout<<"Tuning speed " << tuning_speed << std::endl;
    int N_eqs = 2*Nphi+5;
    VecDoub res_buf(N_eqs);

    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){//resonator field
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+(Nphi)] = res_IM[i_phi] + noise[i_phi].imag();
    }
    res_RE[Nphi] = In_val_RE[Nphi];//CCW resonator field
    res_IM[Nphi] = In_val_IM[Nphi];
    
    res_RE[Nphi+1] = In_val_RE[Nphi+1];//Laser field
    res_IM[Nphi+1] = In_val_IM[Nphi+1];
    
    res_RE[Nphi+2] = In_val_RE[Nphi+2];//Carrier density
    res_IM[Nphi+2] = In_val_IM[Nphi+2];
    
    res_buf[2*Nphi] = In_val_RE[Nphi];//CCW resonator field
    res_buf[2*Nphi + 1] = In_val_IM[Nphi];

    res_buf[2*Nphi+2] = In_val_RE[Nphi+1];//Laser field
    res_buf[2*Nphi+3] = In_val_IM[Nphi+1];


    res_buf[2*Nphi+4] = In_val_RE[Nphi+2];//Carrier density

    Output out;
    rhs_pseudo_spectral_sil_lle lle(Nphi, Dint, detuning[0], kappa_inj, kappa, kappa_sc, kappa_laser, N0, e, I_laser, zeta, gamma, a, V, alpha, eta, coupling_phase, g0, tuning_speed  );
    t_step_ini=t0;
    t_step_end = t1;
    for (int i_det=0; i_det<Ndet; i_det++){
        //lle.det = detuning[i_det];
        noise=WhiteNoise(noise_amp,Nphi);
        Odeint<StepperDopr853<rhs_pseudo_spectral_sil_lle> > ode(res_buf,t_step_ini,t_step_end,atol,rtol,dt,dtmin,out,lle);
        ode.integrate();
        for (int i_phi=0; i_phi<Nphi; i_phi++){//Resonator field
            res_RE[i_det*(Nphi+3)+i_phi] = res_buf[i_phi];
            res_IM[i_det*(Nphi+3)+i_phi] = res_buf[i_phi+Nphi];
            res_buf[i_phi] += noise[i_phi].real();
            res_buf[i_phi+Nphi] += noise[i_phi].imag();
        }
        res_RE[i_det*(Nphi+3)+Nphi] = res_buf[2*Nphi];//CCW resonator field
        res_IM[i_det*(Nphi+3)+Nphi] = res_buf[2*Nphi+1];//

        res_RE[i_det*(Nphi+3)+Nphi+1] = res_buf[2*Nphi+2];//Laser field
        res_IM[i_det*(Nphi+3)+Nphi+1] = res_buf[2*Nphi+3];//

        res_RE[i_det*(Nphi+3)+Nphi+2] = res_buf[2*Nphi+4];//Carrier density
        res_IM[i_det*(Nphi+3)+Nphi+2] = 0.;//Carrier density


        printProgress((i_det+1.)/Ndet);
        t_step_ini+=t1;
        t_step_end+=t1;
        //std::cout<<t_step_ini<< " " << t_step_end << " ";
        //std::this_thread::sleep_for(30ms);


    }
    std::cout<<detuning[Ndet-1]-detuning[0] << " "  << (t_step_end-t1)*tuning_speed << " " << tuning_speed*Tmax << " ";
    delete [] noise;
//    delete [] res_buf;
    std::cout<<"Step adaptative Dopri853 from NR3 is finished\n";
}


void* Propagate_PseudoSpectralSAM_Raman(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double *tau_r_mu, const double* Dint, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM)
    
{
    
    std::cout<<"Pseudo Spectral Step adaptative Dopri853 with Raman from NR3 is running\n";
    std::complex<double>* noise = new (std::nothrow) std::complex<double>[Nphi];
    const double t0=0., t1=(Nt-1)*dt, dtmin=0.;
    double *f = new(std::nothrow) double[2*Nphi];
    VecDoub res_buf(2*Nphi);


    noise=WhiteNoise(noise_amp,Nphi);
    for (int i_phi = 0; i_phi<Nphi; i_phi++){
        res_RE[i_phi] = In_val_RE[i_phi];
        res_IM[i_phi] = In_val_IM[i_phi];
        res_buf[i_phi] = res_RE[i_phi] + noise[i_phi].real();
        res_buf[i_phi+Nphi] = res_IM[i_phi] + noise[i_phi].imag();
        f[i_phi] = Re_F[i_phi];
        f[i_phi+Nphi] = Im_F[i_phi];
    }
    std::cout<<"In val_RE = " << In_val_RE[0]<<std::endl;

    Output out;
    rhs_pseudo_spectral_lle_w_raman lle(Nphi, Dint, detuning[0],f,tau_r_mu);
    
    for (int i_det=0; i_det<Ndet; i_det++){
        lle.det = detuning[i_det];
        noise=WhiteNoise(noise_amp,Nphi);
        Odeint<StepperDopr853<rhs_pseudo_spectral_lle_w_raman> > ode(res_buf,t0,t1,atol,rtol,dt,dtmin,out,lle);
        ode.integrate();
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            res_RE[i_det*Nphi+i_phi] = res_buf[i_phi];
            res_IM[i_det*Nphi+i_phi] = res_buf[i_phi+Nphi];
            res_buf[i_phi] += noise[i_phi].real();
            res_buf[i_phi+Nphi] += noise[i_phi].imag();

        }
        printProgress((i_det+1.)/Ndet);

    }
    delete [] noise;
    delete [] f;
//    delete [] res_buf;
    std::cout<<"Pseudo Spectral Step adaptative Dopri853 with Raman from NR3 is finished\n";
}
