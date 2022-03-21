# PyCORE - is a fast numerical solver for coupled resonators.

### Install gcc-7 compiler
~~~
sudo apt install build-essential
~~~
~~~
gcc --version
~~~
~~~
sudo apt-get install -y gcc-7
~~~
~~~
sudo apt install g++-7
~~~

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


### Compile LLE lib
~~~
g++-7 -fPIC -shared -lfftw3 -lm -O3 -o lib_lle_core.so lle_core.cpp - compile LLE lib
~~~


