#ifndef _BOOST_CROW_CORE_HPP_
#define _BOOST_CROW_CORE_HPP_
#include "boost_lle_core.hpp"

#ifdef  __cplusplus
extern "C" {
#endif

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

    rhs_pseudo_spectral_crow(int Nphii, int Ncrowi, double deti, const double* fi, const double* Dinti, const double* phii, Doub dphii, const double* Ji, const double* kappai, double kappa0i, const double* deltai)
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
        Ncrow = Ncrowi;
        det = crow.det*2./kappa0;
        dphi = crow.dphi;
        J = new (std::nothrow) double[Ncrow-1];
        kappa = new (std::nothrow) double[Ncrow];
        delta = new (std::nothrow) double[Ncrow];
        phi = new (std::nothrow) double[Nphi];
        f = new (std::nothrow) double[2*Nphi*Ncrow];
        Dint = new (std::nothrow) double[Nphi*Ncrow];
        //DispTerm = new (std::nothrow) double[2*Nphi*Ncrow];
        for (int index = 0; index < Ncrow*Nphi; index++){
            Dint[index] = crow.Dint[index]*2./kappa0;
        }

        for (int i_phi = 0; i_phi<2*Nphi*Ncrow; i_phi++){
            f[i_phi] = crow.f[i_phi];
        }
        for (int i_phi = 0; i_phi<Nphi; i_phi++){
            phi[i_phi] = crow.phi[i_phi];
        }
        for (int i_crow = 0; i_crow<Ncrow; i_crow++){
            kappa[i_crow] = crow.kappa[i_crow]/kappa0;
            delta[i_crow] = crow.delta[i_crow]*2./kappa0;
        }
        for (int i_crow = 0; i_crow<Ncrow-1; i_crow++){
            J[i_crow] = crow.J[i_crow]*2./kappa0;
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

};


#ifdef  __cplusplus
}
#endif

#endif
