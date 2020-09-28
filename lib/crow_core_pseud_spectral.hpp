#ifndef _CROW_CORE_HPP_
#define _CROW_CORE_HPP_
#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <random>
#include <cstdio>
#include <fstream>

#include "./../../NR/NR_C301/code/nr3.h"
#include "./../../NR/NR_C301/code/stepper.h"
#include "./../../NR/NR_C301/code/stepperdopr853.h"
#include "./../../NR/NR_C301/code/odeint.h"


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#ifdef  __cplusplus
extern "C" {
#endif

struct rhs_crow_pseudo_spectral{
    Int Nphi, Ncrow;
    Doub det,  dphi, kappa0;
    double *kappa;
    double *J;
    double* phi;
    double* Dint;
    double* f;
    Complex i=1i;
    
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_plan plan_disp_spectrum_2_direct;

    fftw_complex *buf_direct;
    fftw_complex *buf_spectrum;
    fftw_complex *buf_disp_direct;
    fftw_complex *buf_disp_spectrum;

    rhs_crow_pseudo_spectral(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* Dinti, const double* phii, Doub dphii, const double* Ji, const double* kappai, Doub kappa0i)
    {
        std::cout<<"Initializing CROW\n";
        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        det = deti*2./kappa0;
        dphi = dphii;
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];
        
        Dint = new (std::nothrow) double[Ncrow*Nphi];
        
        for (int i_phi = 0; i_phi<Nphi*Ncrow; i_phi++){
            Dint[i_phi] = Dinti[i_phi]*2.0/kappa0;
        }

        for (int i_phi = 0; i_phi<2*Nphi*Ncrow; i_phi++){
            f[i_phi] = fi[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = phii[i_phi];
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            kappa[i_crow] = kappai[i_crow]/kappa0;
        }
        for (int i_crow = 0; i_crow<Ncrow-1; i_crow++){
            J[i_crow] = Ji[i_crow]*2./kappa0;
        }

        buf_direct = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nphi);
        buf_spectrum = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nphi);
        buf_disp_direct = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nphi);
        buf_disp_spectrum = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Nphi);
        
        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_ESTIMATE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_ESTIMATE);
        plan_disp_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_disp_spectrum,buf_disp_direct, FFTW_FORWARD, FFTW_ESTIMATE);


        std::cout<<"CROW is initialized\n";
    }
    ~rhs_crow_pseudo_spectral()
    {
        
        
        delete [] phi;
        delete [] kappa;
        delete [] f;
        delete [] J;
        delete [] Dint;
        
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
        fftw_destroy_plan(plan_disp_spectrum_2_direct);
        free (buf_direct);
        
        free (buf_spectrum);
        free (buf_disp_direct);
        free (buf_disp_spectrum);
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
        
            for (int i_phi = 0; i_phi<Nphi; i_phi++){
                buf_direct[i_phi][0] = y[i_crow*2*Nphi+i_phi];
                buf_direct[i_phi][1] = y[i_crow*2*Nphi+i_phi+Nphi];
            }
            fftw_execute(plan_direct_2_spectrum);

            for (int i_phi = 0; i_phi<Nphi; i_phi++){
                buf_disp_spectrum[i_phi][0] = -Dint[i_crow*Nphi+i_phi]*buf_spectrum[i_phi][0];
                buf_disp_spectrum[i_phi][1] = -Dint[i_crow*Nphi+i_phi]*buf_spectrum[i_phi][1];
            }

            fftw_execute(plan_disp_spectrum_2_direct);

        

            for (int i_phi = 0; i_phi<Nphi; i_phi++){

                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det)  - buf_disp_direct[i_phi][1]/Nphi  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det)  + buf_disp_direct[i_phi][0]/Nphi +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

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
        //std::cout<<y[2*Nphi*Ncrow-1] << " ";
    }
    
};


void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);

void* PropagateSAM_PSEUDO_SPECTRAL(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double* J, const double *phi, const double* Dint,   const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);

#ifdef  __cplusplus
}
#endif

#endif  
