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
    Int Nphi, Ncrow, Ncell;
    Doub det,  dphi, kappa0;
    double *kappa;
    double *delta;
    double *J;
    double *Bus_J;
    double *Bus_Phase;
    double **Bus_coupling_Re;
    double **Bus_coupling_Im;
    double* phi;
    double* DispTerm;
    double* f;
    double *d2;
    Complex i=1i;
    rhs_crow(Int Nphii, Int Ncrowi, Doub deti, const double* fi, const double* d2i, const double* phii, Doub dphii, const double* Ji, const double* Bus_Ji, const double* Bus_Phasei ,const double* kappai, Doub kappa0i, const double* deltai)
    {
        
        std::cout<<"Initializing CROW\n";
        
        double phase=0.;

        kappa0 = kappa0i;
        Nphi = Nphii;
        Ncrow = Ncrowi;
        Ncell = (Ncrowi+1)/2;
        det = deti*2./kappa0;
        dphi = dphii;
        d2 = new (std::nothrow) double[Ncrow];
        J = new (std::nothrow) double[Ncrow-1];
        
        Bus_J = new (std::nothrow) double[Ncell];
        Bus_Phase = new (std::nothrow) double[Ncell];
        
        Bus_coupling_Re = new (std::nothrow) double*[Ncrow];
        Bus_coupling_Im = new (std::nothrow) double*[Ncrow];
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            Bus_coupling_Re[i_crow] = new (std::nothrow) double[Ncrow];
            Bus_coupling_Im[i_crow] = new (std::nothrow) double[Ncrow];

        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            for (int j_crow = 0; j_crow<Ncrow; j_crow++){
                Bus_coupling_Re[i_crow][j_crow] = 0.;
                Bus_coupling_Im[i_crow][j_crow] = 0.;
            }
        }
        
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

        for (int i_cell = 0; i_cell<Ncell; i_cell++){
            Bus_J[i_cell] = Bus_Ji[i_cell]*2./kappa0;
            Bus_Phase[i_cell] = Bus_Phasei[i_cell];
        }
        
        for (int i_crow = 2; i_crow<Ncrow; i_crow+=2){
            for (int j_crow = 0; j_crow=i_crow-1; j_crow+=2){
                phase =0;
                for (int k_crow = j_crow; k_crow=i_crow-1; k_crow+=2){
                    phase+= Bus_Phase[k_crow/2];
                }
                Bus_coupling_Re[i_crow][j_crow] = Bus_J[j_crow/2]*cos(phase);
                Bus_coupling_Im[i_crow][j_crow] = Bus_J[j_crow/2]*sin(phase);
            }
        }
        std::cout<<"snake CROW is initialized\n";
    }
    ~rhs_crow()
    {
        delete [] phi;
        delete [] kappa;
        delete [] delta;
        delete [] d2;
        delete [] f;
        delete [] J;
        delete [] Bus_J;
        delete [] Bus_Phase;

        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            delete [] Bus_coupling_Re[i_crow];
            delete [] Bus_coupling_Im[i_crow];
        }
        delete [] Bus_coupling_Re;
        delete [] Bus_coupling_Im;

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
        
        //Coupling via the bus waveguide
        for (int i_crow = 2; i_crow<Ncrow; i_crow+=2){
            for (int j_crow = 0; j_crow<i_crow; i_crow+=2){
                for (int i_phi = 0; i_phi<Nphi; i_phi++){
                        dydx[i_crow*2*Nphi+i_phi]+= Bus_coupling_Re[i_crow][j_crow]*dydx[i_crow*2*Nphi+i_phi] ;
                        dydx[i_crow*2*Nphi+i_phi+Nphi]+= -Bus_coupling_Im[i_crow][j_crow]*dydx[i_crow*2*Nphi+i_phi+Nphi] ;
                }
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
void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double* kappa, const double kappa0, const double *Delta ,const double* J, const double* Bus_J, const double* Bus_Phase, const double *phi,  const double* d2, const int Ndet, const int Nt, const double dt,const double atol, const double rtol, const int Nphi, const int Ncrow, double noise_amp, double* res_RE, double* res_IM);


#ifdef  __cplusplus
}
#endif

#endif  
