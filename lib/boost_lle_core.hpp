#ifndef _BOOST_LLE_CORE_HPP_
#define _BOOST_LLE_CORE_HPP_
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
struct rhs_pseudo_spectral_lle{
    int Nphi;
    double det, dphi, J;
    double* Dint;
    double* phi;
    //double* DispTerm;
    double* f;
    double buf_re, buf_im;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;

    rhs_pseudo_spectral_lle(int Nphii, const double* Dinti, double deti, const double* fi, const double* phii, double dphii, double Ji)
    {
        //std::cout<<"Initialization started\n";
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
        
        //std::cout<<"Initialization succesfull\n";
    }

    rhs_pseudo_spectral_lle( rhs_pseudo_spectral_lle& lle)
    {
        //std::cout<<"Copying started\n";
        Nphi = lle.Nphi;
        det = lle.det;
        J = lle.J;
        dphi = lle.dphi;
        Dint = new (std::nothrow) double[Nphi];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi];
        //DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = lle.Dint[i_phi];
            phi[i_phi] = lle.phi[i_phi];
            f[i_phi] = lle.f[i_phi];
            f[Nphi+i_phi] = lle.f[Nphi+i_phi];
        }
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));

        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_EXHAUSTIVE);
       
        //std::cout<<"Copying succesfull\n";

    }
    ~rhs_pseudo_spectral_lle()
    {
        //std::cout<<"Deleting started\n";
        delete [] Dint;
        delete [] phi;
        delete [] f;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
        //std::cout<<"Deleting succesfull\n";
    }
    void operator() ( const state_type &y, state_type &dydx, const double x) {
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
void printProgress (double percentage);
void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt    ,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);

#ifdef  __cplusplus
}
#endif

#endif
