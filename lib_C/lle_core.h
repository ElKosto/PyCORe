#ifndef _LLE_CORE_H_
#define _LLE_CORE_H_
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <complex.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#ifdef  __cplusplus
extern "C" {
#endif

void printProgress (double percentage);
complex* WhiteNoise(const double amp, const int Nphi);
void* PropagateSS(double* In_val_RE, double* In_val_IM,const double f,  const double *detuning, const double J, const double *phi, const double* Dint, const int Ndet, const int Nt, const double dt, const int Nphi, double noise_amp, double* res_RE, double* res_IM);

#ifdef  __cplusplus
}
#endif

#endif  /* _LLE_CORE_H_ */
