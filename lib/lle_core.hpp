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
#include <thread>

#include "NR/code/nr3.h"
#include "NR/code/stepper.h"
#include "NR/code/stepperdopr853.h"
#include "NR/code/stepperdopr5.h"
#include "NR/code/odeint.h"


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
struct rhs_pseudo_spectral_sil_lle{
    Int Nphi;
    Doub det ;
    Doub N0, I_laser, e, gamma, kappa, a, V, alpha, coupling_phase, power, psi_0_re, psi_0_im, g0, kappa_sc, kappa_inj, eta, kappa_laser, zeta;
    double* Dint;
    //double* DispTerm;
    double buf_re, buf_im;
    double tuning_speed;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;

    rhs_pseudo_spectral_sil_lle(Int Nphii, const double* Dinti, Doub deti, Doub kappa_inji, Doub kappai, Doub kappa_sci , Doub kappa_laseri, Doub N0i, Doub ei, Doub I_laseri, Doub zetai, Doub gammai, Doub ai, Doub Vi, Doub alphai, Doub etai, Doub coupling_phasei, Doub g0i, Doub tuning_speedi )
    {
        std::cout<<"Initialization started\n";
        Nphi = Nphii;
        det = deti;
        tuning_speed = tuning_speedi;
        kappa = kappai;
        kappa_inj = kappa_inji*2/kappa;
        kappa_sc= kappa_sci*2/kappa;
        kappa_laser = kappa_laseri/kappa;
        N0 = N0i;
        eta = etai;
        e = ei;
        g0 = g0i;
        I_laser = I_laseri;
        zeta = zetai*2/kappa;
        gamma = gammai;
        a = ai;
        V = Vi;
        alpha = alphai;
        coupling_phase = coupling_phasei;
        Dint = new (std::nothrow) double[Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
        }
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        power = 0; 

        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        std::cout<<"Initialization succesfull\n";
    }
    ~rhs_pseudo_spectral_sil_lle()
    {
        delete [] Dint;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        //std::cout<<x<<' ';
        //std::this_thread::sleep_for(300ms);
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_direct[i_phi][0] = y[i_phi];
            buf_direct[i_phi][1] = y[i_phi+Nphi];
        }
        fftw_execute(plan_direct_2_spectrum);
        power = 0.;
        psi_0_re = buf_spectrum[0][0]/(Nphi);///sqrt(Nphi);
        psi_0_im = buf_spectrum[0][1]/(Nphi);
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_re = Dint[i_phi]*buf_spectrum[i_phi][1];
            buf_im =  -Dint[i_phi]*buf_spectrum[i_phi][0];
            buf_spectrum[i_phi][0]= buf_re; 
            buf_spectrum[i_phi][1]= buf_im; 
            power+= (buf_spectrum[i_phi][0]*buf_spectrum[i_phi][0] + buf_spectrum[i_phi][1]*buf_spectrum[i_phi][1])/Nphi/Nphi;
        }
        fftw_execute(plan_spectrum_2_direct);

        for (int i_phi = 0; i_phi<Nphi; i_phi++){

            dydx[i_phi] = -y[i_phi] + (det+tuning_speed*x)*y[i_phi+Nphi]  + buf_direct[i_phi][0]/Nphi - kappa_sc*y[2*Nphi+1]  - (y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi] + 2*(y[2*Nphi]*y[2*Nphi]+y[2*Nphi+1]*y[2*Nphi+1]))*y[i_phi+Nphi] +eta*kappa_inj*sqrt(2*g0/kappa)*(cos(coupling_phase)*y[2*Nphi+2] - sin(coupling_phase)*y[2*Nphi+3] );
            dydx[i_phi+Nphi] = -y[i_phi+Nphi] - (det+tuning_speed*x)*y[i_phi]  + buf_direct[i_phi][1]/Nphi + kappa_sc*y[2*Nphi] +   (y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi] + 2*(y[2*Nphi]*y[2*Nphi]+y[2*Nphi+1]*y[2*Nphi+1]) )*y[i_phi] + eta*kappa_inj*sqrt(2*g0/kappa)*(cos(coupling_phase)*y[2*Nphi+3] + sin(coupling_phase)*y[2*Nphi+2]) ;

        }
        
        //CCW dynamics equation
        dydx[2*Nphi] = -y[2*Nphi] + (det+tuning_speed*x)*y[2*Nphi+1] - kappa_sc*psi_0_im - (y[2*Nphi]*y[2*Nphi] + y[2*Nphi+1]*y[2*Nphi+1] + 2*power)*y[2*Nphi+1];
        dydx[2*Nphi+1] = -y[2*Nphi+1] - (det+tuning_speed*x)*y[2*Nphi] + kappa_sc*psi_0_re + (y[2*Nphi]*y[2*Nphi] + y[2*Nphi+1]*y[2*Nphi+1] + 2*power)*y[2*Nphi];
        //
        
        //Laser dynamics equation
        dydx[2*Nphi+2] = ( a*V/kappa*(y[2*Nphi+4] - N0)-kappa_laser ) * y[2*Nphi+2] + (alpha*(a*V/kappa*(y[2*Nphi+4] - N0)-kappa_laser) - det*0 )*y[2*Nphi+3] + kappa_inj*sqrt(kappa/2/g0)*(cos(coupling_phase)*y[2*Nphi] - sin(coupling_phase)*y[2*Nphi+1] )/eta;
        dydx[2*Nphi+3] = ( a*V/kappa*(y[2*Nphi+4] - N0)-kappa_laser ) * y[2*Nphi+3] - (alpha*(a*V/kappa*(y[2*Nphi+4] - N0)-kappa_laser)  +det*0 )*y[2*Nphi+2] + kappa_inj*sqrt(kappa/2/g0)*(sin(coupling_phase)*y[2*Nphi] + cos(coupling_phase)*y[2*Nphi+1] )/eta;
        //
        
        //Carrier density dynamics equation
        dydx[2*Nphi+4] = 2*I_laser/kappa/e/V - 2*gamma*y[2*Nphi+4]/kappa - 2*a*V/kappa*(y[2*Nphi+4]- N0)*(y[2*Nphi+2]*y[2*Nphi+2] + y[2*Nphi+3]*y[2*Nphi+3]);
    }
    
};
struct rhs_pseudo_spectral_lle_w_raman{
    Int Nphi;
    Doub det;
    double* Dint;
    double* tau_r_mu;
    //double* DispTerm;
    double* f;
    double buf_re, buf_im;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;
    
    fftw_plan plan_direct_2_spectrum_sq;
    fftw_plan plan_spectrum_2_direct_sq;
    fftw_complex *buf_spectrum_sq;
    fftw_complex *buf_direct_sq;

    rhs_pseudo_spectral_lle_w_raman(Int Nphii, const double* Dinti, Doub deti, const double* fi, const double* tau_r_mui)
    {
        std::cout<<"Initialization started\n";
        Nphi = Nphii;
        det = deti;
        Dint = new (std::nothrow) double[Nphi];
        tau_r_mu = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi];
        //DispTerm = new (std::nothrow) double[2*Nphi];
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            Dint[i_phi] = Dinti[i_phi];
            tau_r_mu[i_phi] = tau_r_mui[i_phi];
            f[i_phi] = fi[i_phi];
            f[Nphi+i_phi] = fi[Nphi+i_phi];
        }
        buf_direct = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        buf_direct_sq = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        buf_spectrum_sq = (fftw_complex *) malloc(Nphi*sizeof(fftw_complex));
        
        plan_direct_2_spectrum = fftw_plan_dft_1d(Nphi, buf_direct,buf_spectrum, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct = fftw_plan_dft_1d(Nphi, buf_spectrum,buf_direct, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        
        plan_direct_2_spectrum_sq = fftw_plan_dft_1d(Nphi, buf_direct_sq,buf_spectrum_sq, FFTW_BACKWARD, FFTW_EXHAUSTIVE);
        plan_spectrum_2_direct_sq = fftw_plan_dft_1d(Nphi, buf_spectrum_sq,buf_direct_sq, FFTW_FORWARD, FFTW_EXHAUSTIVE);
        
        std::cout<<"Initialization succesfull\n";
    }
    ~rhs_pseudo_spectral_lle_w_raman()
    {
        delete [] Dint;
        delete [] tau_r_mu;
        delete [] f;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
        free(buf_direct_sq);
        free(buf_spectrum_sq);
        fftw_destroy_plan(plan_direct_2_spectrum_sq);
        fftw_destroy_plan(plan_spectrum_2_direct_sq);
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        for (int i_phi=0; i_phi<Nphi; i_phi++){
            buf_direct[i_phi][0] = y[i_phi];
            buf_direct[i_phi][1] = y[i_phi+Nphi];

            buf_direct_sq[i_phi][0] = buf_direct[i_phi][0]*buf_direct[i_phi][0] + buf_direct[i_phi][1]*buf_direct[i_phi][1];
            buf_direct_sq[i_phi][1] = 0.;
        }
        fftw_execute(plan_direct_2_spectrum);
        fftw_execute(plan_direct_2_spectrum_sq);
        for (int i_phi=0; i_phi<Nphi; i_phi++){

            buf_re = tau_r_mu[i_phi]*buf_spectrum_sq[i_phi][1];
            buf_im = -tau_r_mu[i_phi]*buf_spectrum_sq[i_phi][0];
            buf_spectrum_sq[i_phi][0] = buf_re;
            buf_spectrum_sq[i_phi][1] = buf_im;

            buf_re = Dint[i_phi]*buf_spectrum[i_phi][1];
            buf_im =  -Dint[i_phi]*buf_spectrum[i_phi][0];
            
            buf_spectrum[i_phi][0]= buf_re; 
            buf_spectrum[i_phi][1]= buf_im; 
        }
        fftw_execute(plan_spectrum_2_direct);
        fftw_execute(plan_spectrum_2_direct_sq);

        for (int i_phi = 0; i_phi<Nphi; i_phi++){

            dydx[i_phi] = -y[i_phi] + det*y[i_phi+Nphi]  + buf_direct[i_phi][0]/Nphi  - (y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi+Nphi] + f[i_phi] +y[i_phi+Nphi]*buf_direct_sq[i_phi][0]/Nphi;
            dydx[i_phi+Nphi] = -y[i_phi+Nphi] - det*y[i_phi]  + buf_direct[i_phi][1]/Nphi+(y[i_phi]*y[i_phi]+y[i_phi+Nphi]*y[i_phi+Nphi])*y[i_phi] + f[i_phi+Nphi] - y[i_phi]*buf_direct_sq[i_phi][0]/Nphi;
        }
    }
    
};
void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);

void SaveData( std::complex<double> **A, const double *detuning, const double *phi, const int Ndet, const int Nphi);

void* PropagateSS(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

void* PropagateThermalSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double t_th, const double kappa, const double n2, const double n2t, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

void* Propagate_SiL_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, const double *detuning, const double kappa, const double kappa_laser, const double kappa_sc, const double kappa_inj, const double coupling_phase, const double g0, const double alpha, const double gamma, const double V, const double a, const double e, const double N0, const double eta, const double I_laser, const double zeta , const double* Dint, const int Ndet, const double Tmax, const double Tstep, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

void* Propagate_PseudoSpectralSAM_Raman(double* In_val_RE, double* In_val_IM, double* Re_F, double* Im_F,  const double *detuning, const double *tau_r_mu, const double* Dint, const int Ndet, const int Nt, const double dt,  const double atol, const double rtol, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

#ifdef  __cplusplus
}
#endif

#endif  
