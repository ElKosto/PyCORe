#ifndef _LLE_CORE_HPP_
#define _LLE_CORE_HPP_
#include <iostream>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <random>
#include <cstdio>
#include <fstream>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#ifdef  __cplusplus
extern "C" {
#endif

void printProgress (double percentage);
std::complex<double>* WhiteNoise(const double amp, const int Nphi);
void SaveData( std::complex<double> **A, const double *detuning, const double *phi, const int Ndet, const int Nphi);
void* PropagateSS(double* In_val_RE, double* In_val_IM, double* Re_f, double *Im_F,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

#ifdef  __cplusplus
}
#endif

#endif  
