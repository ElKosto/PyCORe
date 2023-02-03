# PyCORE - is a fast numerical solver for coupled resonators.


### Make sure to have gcc or analgue compiler installed


### Installation of fftw
[Download FFTW](http://www.fftw.org/download.html)
In a folder with FFTW print in the terminal:
~~~
 ./configure
 ~~~
 ~~~
make
~~~
~~~
make install
~~~
~~~
sudo make install
~~~

### Install Anaconda and git software

[Install Anaconda ](https://docs.anaconda.com/anaconda/install/linux/)
[Install .git software ](https://tutorialforlinux.com/2020/07/17/step-by-step-gitahead-ubuntu-20-04-installation-guide/2/ )
[Clone PyCore](https://github.com/ElKosto/PyCORe/tree/PyCORe++)

### To compile the solver with Numerical Recipes (proprietary, http://numerical.recipes/) or odeint (https://headmyshoulder.github.io/odeint-v2/index.html) from Boost (opensoft, https://www.boost.org/), put the folder with the library to an approprite folder (e.g., /usr/local/include/). Running 'cpp -v' in your terminal may help to locate the appropriate folder.  

### Compile LLE lib (all the commands available in lib/README)
~~~
g++ -fPIC -shared -lfftw3 -lm -O3 -o lib_lle_core.so lle_core.cpp 
or 
g++ -std=c++0x -fPIC -shared -lfftw3 -lm -O3 -o lib_boost_lle_core.so boost_lle_core.cpp

~~~


