#ifndef _BOOST_CROW_CORE_HPP_
#define _BOOST_CROW_CORE_HPP_
#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <random>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <unistd.h>
#include <thread>
#include <vector>

#include <boost/numeric/odeint.hpp>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60


#ifdef  __cplusplus
extern "C" {
#endif
typedef std::vector< double > state_type;
struct rhs_pseudo_spectral_crow{
    int Nphi, Ncrow;
    double det,  dphi, kappa0;
    double *kappa;
    double *delta;
    double *J;
    double* phi;
    double* Dint;
    double buf_re, buf_im;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;

    double* f;

    rhs_pseudo_spectral_crow(int Nphii, int Ncrowi, double deti, const double* fi, const double* Dinti, const double* phii, double dphii, const double* Ji, const double* kappai, double kappa0i, const double* deltai)
    {
        //std::cout<<"Initializing pseudo spectral CROW\n";
        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        det = deti*2./kappa0;
        dphi = dphii;
        //d2 = new (std::nothrow) double[Ncrow];
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        delta = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];
        Dint = new (std::nothrow) double[Nphi*Ncrow];
        //DispTerm = new (std::nothrow) double[2*Nphi*Ncrow];
        for (int index = 0; index < Ncrow*Nphi; index++){
            Dint[index] = Dinti[index]*2./kappa0;
        }

        for (int i_phi = 0; i_phi<2*Nphi*Ncrow; i_phi++){
            f[i_phi] = fi[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = phii[i_phi];
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            kappa[i_crow] = kappai[i_crow]/kappa0;
            delta[i_crow] = deltai[i_crow]*2./kappa0;
        }
        for (int i_crow = 0; i_crow<Ncrow-1; i_crow++){
            J[i_crow] = Ji[i_crow]*2./kappa0;
        }
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));

        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_EXHAUSTIVE);

        //std::cout<<"pseudo spetral CROW is initialized\n";
    }    

    rhs_pseudo_spectral_crow( rhs_pseudo_spectral_crow& crow)
    {
        kappa0 = crow.kappa0;
        Nphi = crow.Nphi;
        Ncrow = crow.Ncrow;
        det = crow.det;
        dphi = crow.dphi;
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        delta = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];
        Dint = new (std::nothrow) double[Nphi*Ncrow];
        //DispTerm = new (std::nothrow) double[2*Nphi*Ncrow];
        for (int index = 0; index < Ncrow*Nphi; index++){
            Dint[index] = crow.Dint[index];
        }

        for (int i_phi = 0; i_phi<2*Nphi*Ncrow; i_phi++){
            f[i_phi] = crow.f[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = crow.phi[i_phi];
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            kappa[i_crow] = crow.kappa[i_crow];
            delta[i_crow] = crow.delta[i_crow];
        }
        for (int i_crow = 0; i_crow<Ncrow-1; i_crow++){
            J[i_crow] = crow.J[i_crow];
        }
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));

        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        
    }
    ~rhs_pseudo_spectral_crow()
    {
        delete [] phi;
        delete [] kappa;
        delete [] delta;
        delete [] Dint;
        delete [] f;
        delete [] J;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
    }

    void operator() (const state_type &y, state_type &dydx, const double x) {

        
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){


            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf_direct[i_phi][0] = y[i_crow*2*Nphi+i_phi];
                buf_direct[i_phi][1] = y[i_crow*2*Nphi+i_phi+Nphi];
            }   
            fftw_execute(plan_direct_2_spectrum);
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf_re = Dint[i_crow*Nphi+i_phi]*buf_spectrum[i_phi][1];
                buf_im =  -Dint[i_crow*Nphi+i_phi]*buf_spectrum[i_phi][0];
                buf_spectrum[i_phi][0]= buf_re;
                buf_spectrum[i_phi][1]= buf_im;

            }   
            fftw_execute(plan_spectrum_2_direct);
            for (int i_phi = 0; i_phi<Nphi; i_phi++){

                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det+delta[i_crow])  +buf_direct[i_phi][0]/Nphi  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det+delta[i_crow])  +buf_direct[i_phi][1]/Nphi +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

            }
        }
        int i_crow = 0;
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            dydx[i_crow*2*Nphi+i_phi]+= -(J[i_crow])*(y[(i_crow+1)*2*Nphi+i_phi+Nphi]);
            dydx[i_crow*2*Nphi+i_phi+Nphi]+= (J[i_crow])*(y[(i_crow+1)*2*Nphi+i_phi]);

        }
        i_crow = Ncrow-1;
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            dydx[i_crow*2*Nphi+i_phi]+=  -(J[i_crow-1])*(y[(i_crow-1)*2*Nphi+i_phi+Nphi]);
            dydx[i_crow*2*Nphi+i_phi+Nphi]+=  (J[i_crow-1])*(y[(i_crow-1)*2*Nphi+i_phi]);

        }

        for (int i_crow = 1; i_crow<Ncrow-1; i_crow++){
            for (int i_phi = 0; i_phi<Nphi; i_phi++){
                    dydx[i_crow*2*Nphi+i_phi]+= -(J[i_crow])*(y[(i_crow+1)*2*Nphi+i_phi+Nphi]) -(J[i_crow-1])*(y[(i_crow-1)*2*Nphi+i_phi+Nphi]);
                    dydx[i_crow*2*Nphi+i_phi+Nphi]+= (J[i_crow])*(y[(i_crow+1)*2*Nphi+i_phi])+ (J[i_crow-1])*(y[(i_crow-1)*2*Nphi+i_phi]);
            }
        }
    }    

};

void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double *Delta ,const double* J, const double *phi,  const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);



#ifdef  __cplusplus
}
#endif

#endif
