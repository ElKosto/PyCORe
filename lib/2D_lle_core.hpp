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
    Int Nphi, Ntheta;
    Doub det,  dphi, dtheta;
    double J;
    double* phi;//angle coordinate in the resonator
    double* theta;//angle coordinate of the crow as a ring
    double* DispTerm;
    double* f;
    double d2;
    double j2;
    double delta_theta;//2pi/Ncrow
    Complex i=1i;
    rhs_crow(Int Nphii, Int Nthetai, Doub deti, const double* fi, const double d2i, const double* phii, Doub dphii, const double* thetai, Doub dthetai, const double Ji, const double delta_thetai )
    {
        std::cout<<"Initializing Field Theory normalized CROW with PBC\n";
        Nphi = Nphii;
        Ntheta = Nthetai;
        det = deti;
        dphi = dphii;
        dtheta = dthetai;
        delta_theta=delta_thetai;
        d2 = d2i;
        j2 = Ji;
        phi = new (std::nothrow) double[Nphi];
        theta = new (std::nothrow) double[Ntheta];
        f = new (std::nothrow) double[2*Nphi*Ntheta];
        DispTerm = new (std::nothrow) double[2*Nphi*Ntheta];
        for (int i_phi = 0; i_phi<2*Nphi*Ntheta; i_phi++){
            f[i_phi] = fi[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = phii[i_phi];
        }
        for (int i_theta =0; i_theta<Ntheta; i_theta++){
            theta[i_theta] = thetai[i_theta];
        }
        std::cout<<"FT normalized PBC CROW is initialized\n";
    }
    ~rhs_crow()
    {
        delete [] phi;
        delete [] theta;
        delete [] f;
        delete [] DispTerm;
    }
    void operator() (const Doub x, VecDoub &y, VecDoub &dydx) {
        
        for (int i_theta = 0; i_theta<Ntheta; i_theta++){//Dispersion for phi 
            DispTerm[i_theta*2*Nphi+ 0] = (d2)*(y[i_theta*2*Nphi+1] - 2*y[i_theta*2*Nphi+0]+ y[i_theta*2*Nphi+Nphi-1])/dphi/dphi ;
            DispTerm[i_theta*2*Nphi+ Nphi-1] = (d2)*(y[i_theta*2*Nphi+0] - 2*y[i_theta*2*Nphi+Nphi-1]+ y[i_theta*2*Nphi+Nphi-2])/dphi/dphi;

            DispTerm[i_theta*2*Nphi+Nphi] = (d2)*(y[i_theta*2*Nphi+Nphi+1] - 2*y[i_theta*2*Nphi+Nphi]+ y[i_theta*2*Nphi+2*Nphi-1])/dphi/dphi;
            DispTerm[i_theta*2*Nphi+2*Nphi-1] = (d2)*(y[i_theta*2*Nphi+Nphi] - 2*y[i_theta*2*Nphi+2*Nphi-1]+ y[i_theta*2*Nphi+2*Nphi-2])/dphi/dphi;


            for (int i_phi = 1; i_phi<Nphi-1; i_phi++){
                DispTerm[i_theta*2*Nphi+i_phi] = (d2)*(y[i_theta*2*Nphi+i_phi+1] - 2*y[i_theta*2*Nphi+i_phi]+ y[i_theta*2*Nphi+i_phi-1])/dphi/dphi;
                DispTerm[i_theta*2*Nphi+i_phi+Nphi] = (d2)*(y[i_theta*2*Nphi+i_phi+Nphi+1] - 2*y[i_theta*2*Nphi+i_phi+Nphi]+ y[i_theta*2*Nphi+i_phi+Nphi-1])/dphi/dphi;
            }
        }


        for (int i_phi = 0; i_phi<Nphi; i_phi++){//Dispersion for theta 
            // i_theta = 0;
            DispTerm[i_phi]+= delta_theta*j2*(y[2*Nphi+i_phi] - 2*y[i_phi]+ y[(Ntheta-1)*2*Nphi+i_phi])/dtheta/dtheta;
            DispTerm[i_phi+Nphi]+= delta_theta*j2*(y[2*Nphi+i_phi+Nphi] - 2*y[i_phi+Nphi]+ y[(Ntheta-1)*2*Nphi+i_phi+Nphi])/dtheta/dtheta;
            
            // i_theta = Ntheta-1;
            DispTerm[(Ntheta-1)*2*Nphi+i_phi]+= delta_theta*j2*(y[(0)*2*Nphi+i_phi] - 2*y[(Ntheta-1)*2*Nphi+i_phi]+ y[(Ntheta-2)*2*Nphi+i_phi])/dtheta/dtheta;
            DispTerm[(Ntheta-1)*2*Nphi+i_phi+Nphi]+= delta_theta*j2*(y[(0)*2*Nphi+i_phi+Nphi] - 2*y[(Ntheta-1)*2*Nphi+i_phi+Nphi]+ y[(Ntheta-2)*2*Nphi+i_phi+Nphi])/dtheta/dtheta;
            
            for (int i_theta = 1; i_theta<Ntheta-1; i_theta++){
                DispTerm[i_theta*2*Nphi+i_phi]+= delta_theta*j2*(y[(i_theta+1)*2*Nphi+i_phi] - 2*y[i_theta*2*Nphi+i_phi]+ y[(i_theta-1)*2*Nphi+i_phi])/dtheta/dtheta;
                DispTerm[i_theta*2*Nphi+i_phi+Nphi]+= delta_theta*j2*(y[(i_theta)*2*Nphi+i_phi+Nphi] - 2*y[i_theta*2*Nphi+i_phi+Nphi]+ y[(i_theta-1)*2*Nphi+i_phi+Nphi])/dtheta/dtheta;
            }

        }
        
        for (int i_theta = 0; i_theta<Ntheta; i_theta++){
            for (int i_phi = 0; i_phi<Nphi; i_phi++){

                dydx[i_theta*2*Nphi+i_phi] = -y[i_theta*2*Nphi+i_phi] + y[i_theta*2*Nphi+i_phi+Nphi]*(det)  - DispTerm[i_theta*2*Nphi+i_phi+Nphi]  - (y[i_theta*2*Nphi+i_phi]*y[i_theta*2*Nphi+i_phi]+y[i_theta*2*Nphi+i_phi+Nphi]*y[i_theta*2*Nphi+i_phi+Nphi])*y[i_theta*2*Nphi+i_phi+Nphi] + f[i_theta*2*Nphi+i_phi];
                dydx[i_theta*2*Nphi+i_phi+Nphi] = -y[i_theta*2*Nphi+i_phi+Nphi] - y[i_theta*2*Nphi+i_phi]*(det)  + DispTerm[i_theta*2*Nphi+i_phi] +(y[i_theta*2*Nphi+i_phi]*y[i_theta*2*Nphi+i_phi]+y[i_theta*2*Nphi+i_phi+Nphi]*y[i_theta*2*Nphi+i_phi+Nphi])*y[i_theta*2*Nphi+i_phi] + f[i_theta*2*Nphi+i_phi+Nphi];

            }
        }
        //std::cout<<y[2*Nphi*Ntheta-1] << " ";
    }
    
};

void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);
void* PropagateSAM(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* theta, const double delta_theta, const double d2, const double j2, const int Ndet, const int Nt, const double dt, const double atol, const double rtol, const int Nphi, const int Ntheta, double noise_amp, double* res_RE, double* res_IM);


#ifdef  __cplusplus
}
#endif

#endif  
