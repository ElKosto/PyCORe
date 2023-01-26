# PyCORe
This is Python and C-based solver for the Lugiato-Lefever Equation. 

In the main branch we keep only Python code for simulation of nonlnear dynamics in a single   &chi;<sup>(3)</sup> microresonator. This allows to simply use the solver on any machine with installed numpy package.

In the branch PyCore++ we use C-based solver with Python interface. There are two solvers: Step adaptative which uses Numerical recipes 3 library and Split step method which uses fftw library. In order to use it one need to have both on the PC.

A set of simple test scripts is in the tests folder. 
