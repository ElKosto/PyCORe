#ifndef _CROW_CORE_HPP_
#define _CROW_CORE_HPP_
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


#include "./../../NR/NR_C301/code/nr3.h"
#include "./../../NR/NR_C301/code/stepper.h"
#include "./../../NR/NR_C301/code/stepperdopr853.h"
#include "./../../NR/NR_C301/code/odeint.h"


#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#ifdef  __cplusplus
extern "C" {
#endif

struct rhs_crow{
    Int Nphi, Ncrow;
    Doub det,  dphi, kappa0;
    double *kappa;
    double *delta;
    double *J;
    double* phi;
    double* DispTerm;
    double* f;
    double *d2;
    Complex i=1i;
    rhs_crow(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* d2i, const double* phii, Doub dphii, const double* Ji, const double* kappai, Doub kappa0i, const double* deltai)
    {
        std::cout<<"Initializing CROW\n";
        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        det = deti*2./kappa0;
        dphi = dphii;
        d2 = new (std::nothrow) double[Ncrow];
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        delta = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];
        DispTerm = new (std::nothrow) double[2*Nphi*Ncrow];
        for (int i_phi = 0; i_phi<2*Nphi*Ncrow; i_phi++){
            f[i_phi] = fi[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = phii[i_phi];
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            d2[i_crow] = d2i[i_crow]/kappa0;
            kappa[i_crow] = kappai[i_crow]/kappa0;
	    delta[i_crow] = deltai[i_crow]*2./kappa0;
        }
        for (int i_crow = 0; i_crow<Ncrow-1; i_crow++){
            J[i_crow] = Ji[i_crow]*2./kappa0;
        }
        std::cout<<"CROW is initialized\n";
    }
    ~rhs_crow()
    {
        delete [] phi;
        delete [] kappa;
        delete [] delta;
        delete [] d2;
        delete [] f;
        delete [] J;
        delete [] DispTerm;
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            DispTerm[i_crow*2*Nphi+ 0] = (d2[i_crow])*(y[i_crow*2*Nphi+1] - 2*y[i_crow*2*Nphi+0]+ y[i_crow*2*Nphi+Nphi-1])/dphi/dphi;
            DispTerm[i_crow*2*Nphi+ Nphi-1] = (d2[i_crow])*(y[i_crow*2*Nphi+0] - 2*y[i_crow*2*Nphi+Nphi-1]+ y[i_crow*2*Nphi+Nphi-2])/dphi/dphi;

            DispTerm[i_crow*2*Nphi+Nphi] = (d2[i_crow])*(y[i_crow*2*Nphi+Nphi+1] - 2*y[i_crow*2*Nphi+Nphi]+ y[i_crow*2*Nphi+2*Nphi-1])/dphi/dphi;
            DispTerm[i_crow*2*Nphi+2*Nphi-1] = (d2[i_crow])*(y[i_crow*2*Nphi+Nphi] - 2*y[i_crow*2*Nphi+2*Nphi-1]+ y[i_crow*2*Nphi+2*Nphi-2])/dphi/dphi;


            for (int i_phi = 1; i_phi<Nphi-1; i_phi++){
                DispTerm[i_crow*2*Nphi+i_phi] = (d2[i_crow])*(y[i_crow*2*Nphi+i_phi+1] - 2*y[i_crow*2*Nphi+i_phi]+ y[i_crow*2*Nphi+i_phi-1])/dphi/dphi;
                DispTerm[i_crow*2*Nphi+i_phi+Nphi] = (d2[i_crow])*(y[i_crow*2*Nphi+i_phi+Nphi+1] - 2*y[i_crow*2*Nphi+i_phi+Nphi]+ y[i_crow*2*Nphi+i_phi+Nphi-1])/dphi/dphi;
            }
        
        

            for (int i_phi = 0; i_phi<Nphi; i_phi++){

                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det+delta[i_crow])  - DispTerm[i_crow*2*Nphi+i_phi+Nphi]  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det+delta[i_crow])  + DispTerm[i_crow*2*Nphi+i_phi] +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

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
struct rhs_pseudo_spectral_crow{
    Int Nphi, Ncrow;
    Doub det,  dphi, kappa0;
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
    Complex i=1i;
    rhs_pseudo_spectral_crow(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* Dinti, const double* phii, Doub dphii, const double* Ji, const double* kappai, Doub kappa0i, const double* deltai)
    {
        std::cout<<"Initializing pseudo spectral CROW\n";
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

        std::cout<<"speudo spetral CROW is initialized\n";
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
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            //DispTerm[i_crow*2*Nphi+ 0] = (d2[i_crow])*(y[i_crow*2*Nphi+1] - 2*y[i_crow*2*Nphi+0]+ y[i_crow*2*Nphi+Nphi-1])/dphi/dphi;
            //DispTerm[i_crow*2*Nphi+ Nphi-1] = (d2[i_crow])*(y[i_crow*2*Nphi+0] - 2*y[i_crow*2*Nphi+Nphi-1]+ y[i_crow*2*Nphi+Nphi-2])/dphi/dphi;

            //DispTerm[i_crow*2*Nphi+Nphi] = (d2[i_crow])*(y[i_crow*2*Nphi+Nphi+1] - 2*y[i_crow*2*Nphi+Nphi]+ y[i_crow*2*Nphi+2*Nphi-1])/dphi/dphi;
            //DispTerm[i_crow*2*Nphi+2*Nphi-1] = (d2[i_crow])*(y[i_crow*2*Nphi+Nphi] - 2*y[i_crow*2*Nphi+2*Nphi-1]+ y[i_crow*2*Nphi+2*Nphi-2])/dphi/dphi;


            //for (int i_phi = 1; i_phi<Nphi-1; i_phi++){
            //    DispTerm[i_crow*2*Nphi+i_phi] = (d2[i_crow])*(y[i_crow*2*Nphi+i_phi+1] - 2*y[i_crow*2*Nphi+i_phi]+ y[i_crow*2*Nphi+i_phi-1])/dphi/dphi;
            //    DispTerm[i_crow*2*Nphi+i_phi+Nphi] = (d2[i_crow])*(y[i_crow*2*Nphi+i_phi+Nphi+1] - 2*y[i_crow*2*Nphi+i_phi+Nphi]+ y[i_crow*2*Nphi+i_phi+Nphi-1])/dphi/dphi;
            //}
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

                //dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det+delta[i_crow])  - DispTerm[i_crow*2*Nphi+i_phi+Nphi]  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det+delta[i_crow])  +buf_direct[i_phi][0]/Nphi  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                //dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det+delta[i_crow])  + DispTerm[i_crow*2*Nphi+i_phi] +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]
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
        //std::cout<<y[2*Nphi*Ncrow-1] << " ";
    }
    
};
struct rhs_crow_thermal{
    Int Nphi, Ncrow;
    Doub det,  dphi, kappa0;
    double *kappa;
    double *power;
    Doub t_th, n2, n2t;
    double *delta;
    double *J;
    double* phi;
    double* DispTerm;
    double* f;
    double *d2;
    Complex i=1i;
    rhs_crow_thermal(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* d2i, const double* phii, Doub dphii, const double* Ji, const double* kappai, Doub kappa0i, const double* deltai, Doub t_thi, Doub  n2i, Doub n2ti)
    {
        std::cout<<"Initializing CROW with thermal effects\n";
        n2 = n2i;
        n2t = n2ti;
        t_th = t_thi;
        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        det = deti*2./kappa0;
        dphi = dphii;
        d2 = new (std::nothrow) double[Ncrow];
        power = new (std::nothrow) double[Ncrow];
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        delta = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];
        DispTerm = new (std::nothrow) double[2*Nphi*Ncrow];
        for (int i_phi = 0; i_phi<2*Nphi*Ncrow; i_phi++){
            f[i_phi] = fi[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = phii[i_phi];
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            d2[i_crow] = d2i[i_crow]/kappa0;
            kappa[i_crow] = kappai[i_crow]/kappa0;
            delta[i_crow] = deltai[i_crow]*2./kappa0;
        }
        for (int i_crow = 0; i_crow<Ncrow-1; i_crow++){
            J[i_crow] = Ji[i_crow]*2./kappa0;
        }
        std::cout<<"CROW is initialized\n";
    }
    ~rhs_crow_thermal()
    {
        delete [] phi;
        delete [] kappa;
        delete [] delta;
        delete [] d2;
        delete [] power;
        delete [] f;
        delete [] J;
        delete [] DispTerm;
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            DispTerm[i_crow*2*Nphi+ 0] = (d2[i_crow])*(y[i_crow*2*Nphi+1] - 2*y[i_crow*2*Nphi+0]+ y[i_crow*2*Nphi+Nphi-1])/dphi/dphi;
            DispTerm[i_crow*2*Nphi+ Nphi-1] = (d2[i_crow])*(y[i_crow*2*Nphi+0] - 2*y[i_crow*2*Nphi+Nphi-1]+ y[i_crow*2*Nphi+Nphi-2])/dphi/dphi;

            DispTerm[i_crow*2*Nphi+Nphi] = (d2[i_crow])*(y[i_crow*2*Nphi+Nphi+1] - 2*y[i_crow*2*Nphi+Nphi]+ y[i_crow*2*Nphi+2*Nphi-1])/dphi/dphi;
            DispTerm[i_crow*2*Nphi+2*Nphi-1] = (d2[i_crow])*(y[i_crow*2*Nphi+Nphi] - 2*y[i_crow*2*Nphi+2*Nphi-1]+ y[i_crow*2*Nphi+2*Nphi-2])/dphi/dphi;
            
            power[i_crow] = dphi/2*(y[i_crow*2*Nphi+0]*y[i_crow*2*Nphi+0] + y[i_crow*2*Nphi+Nphi+0]*y[i_crow*2*Nphi+Nphi+0] + y[i_crow*2*Nphi+1]*y[i_crow*2*Nphi+1]+y[i_crow*2*Nphi+Nphi+1]*y[i_crow*2*Nphi+Nphi+1]);

            for (int i_phi = 1; i_phi<Nphi-1; i_phi++){
                DispTerm[i_crow*2*Nphi+i_phi] = (d2[i_crow])*(y[i_crow*2*Nphi+i_phi+1] - 2*y[i_crow*2*Nphi+i_phi]+ y[i_crow*2*Nphi+i_phi-1])/dphi/dphi;
                DispTerm[i_crow*2*Nphi+i_phi+Nphi] = (d2[i_crow])*(y[i_crow*2*Nphi+i_phi+Nphi+1] - 2*y[i_crow*2*Nphi+i_phi+Nphi]+ y[i_crow*2*Nphi+i_phi+Nphi-1])/dphi/dphi;
                power[i_crow] += dphi/2*(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi] + y[i_crow*2*Nphi+Nphi+i_phi]*y[i_crow*2*Nphi+Nphi+i_phi] + y[i_crow*2*Nphi+i_phi+1]*y[i_crow*2*Nphi+i_phi+1]+y[i_crow*2*Nphi+Nphi+i_phi+1]*y[i_crow*2*Nphi+Nphi+i_phi+1]);
            }
        
        

            for (int i_phi = 0; i_phi<Nphi; i_phi++){

                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det+delta[i_crow]-y[2*Ncrow*Nphi+i_crow])  - DispTerm[i_crow*2*Nphi+i_phi+Nphi]  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det+delta[i_crow]-y[2*Ncrow*Nphi+i_crow])  + DispTerm[i_crow*2*Nphi+i_phi] +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

            }
            
            dydx[2*Nphi*Ncrow+i_crow] = 2/kappa0/t_th*(n2t/n2/kappa[i_crow]/2/M_PI*power[i_crow]-y[2*Ncrow*Nphi+i_crow]);
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

struct rhs_pseudo_spectral_crow_thermal{
    Int Nphi, Ncrow;
    Doub det,  dphi, kappa0;
    double *kappa;
    double *power;
    Doub t_th, n2, n2t;
    double *delta;
    double *J;
    double* phi;
    double* Dint;
    double* f;
    Complex i=1i;
    double buf_re, buf_im;
    fftw_plan plan_direct_2_spectrum;
    fftw_plan plan_spectrum_2_direct;
    fftw_complex *buf_direct, *buf_spectrum;

    rhs_pseudo_spectral_crow_thermal(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* Dinti, const double* phii, Doub dphii, const double* Ji, const double* kappai, Doub kappa0i, const double* deltai, Doub t_thi, Doub  n2i, Doub n2ti)
    {
        std::cout<<"Initializing CROW with thermal effects\n";
        n2 = n2i;
        n2t = n2ti;
        t_th = t_thi;
        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        det = deti*2./kappa0;
        Dint = new (std::nothrow) double[Nphi*Ncrow];
        dphi = dphii;
        power = new (std::nothrow) double[Ncrow];
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        delta = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];

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
        
        std::cout<<"CROW is initialized\n";
    }
    ~rhs_pseudo_spectral_crow_thermal()
    {
        delete [] phi;
        delete [] kappa;
        delete [] delta;
        delete [] power;
        delete [] f;
        delete [] J;
        free(buf_direct);
        free(buf_spectrum);
        fftw_destroy_plan(plan_direct_2_spectrum);
        fftw_destroy_plan(plan_spectrum_2_direct);
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){

            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf_direct[i_phi][0] = y[i_crow*2*Nphi+i_phi];
                buf_direct[i_phi][1] = y[i_crow*2*Nphi+i_phi+Nphi];
 //               power[i_crow] += dphi/2*(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi] + y[i_crow*2*Nphi+Nphi+i_phi]*y[i_crow*2*Nphi+Nphi+i_phi] + y[i_crow*2*Nphi+i_phi+1]*y[i_crow*2*Nphi+i_phi+1]+y[i_crow*2*Nphi+Nphi+i_phi+1]*y[i_crow*2*Nphi+Nphi+i_phi+1]);
            }
            fftw_execute(plan_direct_2_spectrum);
            for (int i_phi=0; i_phi<Nphi; i_phi++){
                buf_re = Dint[i_crow*Nphi+i_phi]*buf_spectrum[i_phi][1];
                buf_im =  -Dint[i_crow*Nphi+i_phi]*buf_spectrum[i_phi][0];
                power[i_crow] += 2*M_PI*(buf_spectrum[i_phi][0]*buf_spectrum[i_phi][0] + buf_spectrum[i_phi][1]*buf_spectrum[i_phi][1]);
                buf_spectrum[i_phi][0]= buf_re;
                buf_spectrum[i_phi][1]= buf_im;

                
            }
            fftw_execute(plan_spectrum_2_direct);
        
        

            for (int i_phi = 0; i_phi<Nphi; i_phi++){

                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det+delta[i_crow])  +buf_direct[i_phi][0]/Nphi  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];
                dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det+delta[i_crow])  +buf_direct[i_phi][1]/Nphi +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];

            }

            dydx[2*Nphi*Ncrow+i_crow] = 2/kappa0/t_th*(n2t/n2/kappa[i_crow]/2/M_PI*power[i_crow]-y[2*Ncrow*Nphi+i_crow]);
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
void printJ(rhs_crow lle)
{
    for (int i_crow = 0; i_crow<lle.Ncrow-1; i_crow++){
        std::cout<<lle.J[i_crow]<<" ";
    }
    std::cout<<"\n";
};

void printKappa(rhs_crow lle)
{
    for (int i_crow = 0; i_crow<lle.Ncrow; i_crow++){
        std::cout<<lle.kappa[i_crow]<<" ";
    }
    std::cout<<"\n";
};

void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);
void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double *Delta ,const double* J, const double *phi,  const double* d2, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);
void* Propagate_PseudoSpectralSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double *Delta ,const double* J, const double *phi,  const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);
void* PropagateThermalSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double t_th, const double n2, const double n2t, const double *Delta ,const double* J, const double *phi,  const double* d2, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);
void* Propagate_PseudoSpectralThermalSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double t_th, const double n2, const double n2t, const double *Delta ,const double* J, const double *phi,  const double* Dint, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);


#ifdef  __cplusplus
}
#endif

#endif  
