#ifndef _LLE_CORE_HPP_
#define _LLE_CORE_HPP_
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
void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);
void SaveData( std::complex<double> **A, const double *detuning, const double *phi, const int Ndet, const int Nphi);
void* PropagateSS(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const double *d_2_osc, const double FSR, const double kappa, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM);
void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

#ifdef  __cplusplus
}
#endif

#endif  
