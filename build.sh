#!/bin/bash

#export LD_LIBRARY_PATH=/home/dfried/lib/opencv/build/lib/:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH="/home/dfried/local_install/lib/:${LD_LIBRARY_PATH}"
export CC=gcc-5
export CXX=g++-5

source activate m3d3
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="/home/dfried/local_install/" ..
make -j12
