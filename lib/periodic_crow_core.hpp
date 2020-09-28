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

struct rhs_crow{
    Int Nphi, Ncrow;
    Doub det,  dphi, kappa0;
    double *kappa;
    double *J;
    double* phi;
    double* DispTerm;
    double* f;
    double *d2;
    Complex i=1i;
    rhs_crow(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* d2i, const double* phii, Doub dphii, const double* Ji, const double* kappai, Doub kappa0i)
    {
        std::cout<<"Initializing CROW\n";
        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        det = deti*2./kappa0;
        dphi = dphii;
        d2 = new (std::nothrow) double[Ncrow];
        J = new (std::nothrow) double[Ncrow];
        kappa = new (std::nothrow) double[Ncrow];
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
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            J[i_crow] = Ji[i_crow]*2./kappa0;
        }
        std::cout<<"CROW is initialized\n";
    }
    ~rhs_crow()
    {
        delete [] phi;
        delete [] kappa;
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

                dydx[i_crow*2*Nphi+i_phi] = -y[i_crow*2*Nphi+i_phi]*(kappa[i_crow]) + y[i_crow*2*Nphi+i_phi+Nphi]*(det)  - DispTerm[i_crow*2*Nphi+i_phi+Nphi]  - (y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi+Nphi] + f[i_crow*2*Nphi+i_phi];//- J*cos(phi[i_phi])*y[i_phi+Nphi]
                dydx[i_crow*2*Nphi+i_phi+Nphi] = -y[i_crow*2*Nphi+i_phi+Nphi]*(kappa[i_crow]) - y[i_crow*2*Nphi+i_phi]*(det)  + DispTerm[i_crow*2*Nphi+i_phi] +(y[i_crow*2*Nphi+i_phi]*y[i_crow*2*Nphi+i_phi]+y[i_crow*2*Nphi+i_phi+Nphi]*y[i_crow*2*Nphi+i_phi+Nphi])*y[i_crow*2*Nphi+i_phi] + f[i_crow*2*Nphi+i_phi+Nphi];//+ J*cos(phi[i_phi])*y[i_phi]

            }
        }
        int i_crow = 0;
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            dydx[i_crow*2*Nphi+i_phi]+= -(J[i_crow])*(y[(i_crow+1)*2*Nphi+i_phi+Nphi]) - (J[Ncrow-1])*(y[(Ncrow-1)*2*Nphi+i_phi+Nphi]);
            dydx[i_crow*2*Nphi+i_phi+Nphi]+= (J[i_crow])*(y[(i_crow+1)*2*Nphi+i_phi]) + (J[Ncrow-1])*(y[(Ncrow-1)*2*Nphi+i_phi]);
            
        }
        i_crow = Ncrow-1;
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            dydx[i_crow*2*Nphi+i_phi]+=  -(J[i_crow-1])*(y[(i_crow-1)*2*Nphi+i_phi+Nphi]) - (J[i_crow])*(y[(0)*2*Nphi+i_phi+Nphi]);
            dydx[i_crow*2*Nphi+i_phi+Nphi]+=  (J[i_crow-1])*(y[(i_crow-1)*2*Nphi+i_phi]) + (J[i_crow])*(y[(0)*2*Nphi+i_phi]);
            
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
void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double* J, const double *phi,  const double* d2, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);


#ifdef  __cplusplus
}
#endif

#endif  
