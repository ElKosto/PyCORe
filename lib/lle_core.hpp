#ifndef _LLE_CORE_HPP_
#define _LLE_CORE_HPP_
#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <random>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <unistd.h>

#include "./../../NR/NR_C301/code/nr3.h"
#include "./../../NR/NR_C301/code/stepper.h"
#include "./../../NR/NR_C301/code/stepperdopr853.h"
#include "./../../NR/NR_C301/code/odeint.h"


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#ifdef  __cplusplus
extern "C" {
#endif

struct rhs_lle{
    Int Nphi;
    Doub det, d2, dphi, J;
    double* Dint;
    double* phi;
    double* DispTerm;
    double* f;
    Complex i=1i;

    rhs_lle(Int Nphii, const double* Dinti, Doub deti, const double* fi, Doub d2i, const double* phii, Doub dphii, Doub Ji)
    {
        Nphi = Nphii;
        det = deti;
        d2 = d2i;
        J = Ji;
        dphi = dphii;
        Dint = new (std::nothrow) double[Nphi];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi];
        DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
            phi[i_phi] = phii[i_phi];
            f[i_phi] = fi[i_phi];
            f[Nphi+i_phi] = fi[Nphi+i_phi];
        }
    }
    ~rhs_lle()
    {
        delete [] Dint;
        delete [] phi;
        delete [] f;
        delete [] DispTerm;
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {

        DispTerm[0] = d2*(y[1] - 2*y[0]+ y[Nphi-1])/dphi/dphi;
        DispTerm[Nphi-1] = d2*(y[0] - 2*y[Nphi-1]+ y[Nphi-2])/dphi/dphi;

        DispTerm[Nphi] = d2*(y[Nphi+1] - 2*y[Nphi]+ y[2*Nphi-1])/dphi/dphi;
        DispTerm[2*Nphi-1] = d2*(y[Nphi] - 2*y[2*Nphi-1]+ y[2*Nphi-2])/dphi/dphi;


        for (int i_phi = 1; i_phi<Nphi-1; i_phi++){
            DispTerm[i_phi] = d2*(y[i_phi+1] - 2*y[i_phi]+ y[i_phi-1])/dphi/dphi;
            DispTerm[i_phi+Nphi] = d2*(y[i_phi+Nphi+1] - 2*y[i_phi+Nphi]+ y[i_phi+Nphi-1])/dphi/dphi;
        }

        for (int i_phi = 0; i_phi<Nphi; i_phi++){

            dydx[i_phi] = -y[i_phi] + det*y[i_phi+Nphi]  - DispTerm[i_phi+Nphi]  - (y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi+Nphi] + f[i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
            dydx[i_phi+Nphi] = -y[i_phi+Nphi] - det*y[i_phi]  + DispTerm[i_phi]+(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi] + f[i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

        }
    }
    
};
struct rhs_pseudo_spectral_lle{
    Int Nphi;
    Doub det, dphi, J;
    double* Dint;
    double* phi;
    //double* DispTerm;
    double* f;
    Complex i=1i;
    double buf_re, buf_im;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;

    rhs_pseudo_spectral_lle(Int Nphii, const double* Dinti, Doub deti, const double* fi, const double* phii, Doub dphii, Doub Ji)
    {
        std::cout<<"Initialization started\n";
        Nphi = Nphii;
        det = deti;
        J = Ji;
        dphi = dphii;
        Dint = new (std::nothrow) double[Nphi];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi];
        //DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
            phi[i_phi] = phii[i_phi];
            f[i_phi] = fi[i_phi];
            f[Nphi+i_phi] = fi[Nphi+i_phi];
        }
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        std::cout<<"Initialization succesfull\n";
    }
    ~rhs_pseudo_spectral_lle()
    {
        delete [] Dint;
        delete [] phi;
        delete [] f;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_direct[i_phi][0] = y[i_phi];
            buf_direct[i_phi][1] = y[i_phi+Nphi];
        }
        fftw_execute(plan_direct_2_spectrum);
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_re = Dint[i_phi]*buf_spectrum[i_phi][1];
            buf_im =  -Dint[i_phi]*buf_spectrum[i_phi][0];
            buf_spectrum[i_phi][0]= buf_re; 
            buf_spectrum[i_phi][1]= buf_im; 
        }
        fftw_execute(plan_spectrum_2_direct);

        for (int i_phi = 0; i_phi<Nphi; i_phi++){

            dydx[i_phi] = -y[i_phi] + det*y[i_phi+Nphi]  + buf_direct[i_phi][0]/Nphi  - (y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi+Nphi] + f[i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
            dydx[i_phi+Nphi] = -y[i_phi+Nphi] - det*y[i_phi]  + buf_direct[i_phi][1]/Nphi+(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi] + f[i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

        }
    }
    
};
struct rhs_lle_thermal{
    Int Nphi;
    Doub det, d2, dphi, J, t_th, kappa, n2, n2t;
    double* Dint;
    double* phi;
    double* DispTerm;
    double* f;
    Complex i=1i;
    rhs_lle_thermal(Int Nphii, const double* Dinti, Doub deti, const double* fi, Doub d2i, const double* phii, Doub dphii, Doub Ji, Doub t_thi, Doub kappai, Doub n2i, Doub n2ti)
    {
        Nphi = Nphii;
        det = deti;
        d2 = d2i;
        J = Ji;
        t_th = t_thi;
        kappa = kappai;
        n2 = n2i;
        n2t = n2ti;
        dphi = dphii;
        Dint = new (std::nothrow) double[Nphi];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi];
        DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
            phi[i_phi] = phii[i_phi];
            f[i_phi] = fi[i_phi];
            f[Nphi+i_phi] = fi[Nphi+i_phi];
        }
    }
    ~rhs_lle_thermal()
    {
        delete [] Dint;
        delete [] phi;
        delete [] f;
        delete [] DispTerm;
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {

        double power=0.;

        DispTerm[0] = d2*(y[1] - 2*y[0]+ y[Nphi-1])/dphi/dphi;
        DispTerm[Nphi-1] = d2*(y[0] - 2*y[Nphi-1]+ y[Nphi-2])/dphi/dphi;

        DispTerm[Nphi] = d2*(y[Nphi+1] - 2*y[Nphi]+ y[2*Nphi-1])/dphi/dphi;
        DispTerm[2*Nphi-1] = d2*(y[Nphi] - 2*y[2*Nphi-1]+ y[2*Nphi-2])/dphi/dphi;
        
        power=dphi*(y[0]*y[0]+y[Nphi]*y[Nphi] + y[1]*y[1] + y[Nphi+1]*y[Nphi+1])/2;

        for (int i_phi = 1; i_phi<Nphi-1; i_phi++){
            DispTerm[i_phi] = d2*(y[i_phi+1] - 2*y[i_phi]+ y[i_phi-1])/dphi/dphi;
            DispTerm[i_phi+Nphi] = d2*(y[i_phi+Nphi+1] - 2*y[i_phi+Nphi]+ y[i_phi+Nphi-1])/dphi/dphi;
        
            power+=dphi*(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi] + y[i_phi+1]*y[i_phi+1] + y[i_phi+Nphi+1]*y[i_phi+Nphi+1])/2;
        }

        for (int i_phi = 0; i_phi<Nphi; i_phi++){

            dydx[i_phi] = -y[i_phi] + (det-y[2*Nphi])*y[i_phi+Nphi]  - DispTerm[i_phi+Nphi]  - (y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi+Nphi] + f[i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
            dydx[i_phi+Nphi] = -y[i_phi+Nphi] - (det-y[2*Nphi])*y[i_phi]  + DispTerm[i_phi]+(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi] + f[i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

        }
        dydx[2*Nphi] = 2/kappa/t_th*(n2t/n2*1/2/M_PI*power-y[2*Nphi]);
    }
    
};
void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);
void SaveData( std::complex<double> **A, const double *detuning, const double *phi, const int Ndet, const int Nphi);
void* PropagateSS(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM);
void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);
void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);
void* PropagateThermalSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double t_th, const double kappa, const double n2, const double n2t, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

#ifdef  __cplusplus
}
#endif

#endif  
