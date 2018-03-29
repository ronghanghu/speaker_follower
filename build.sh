#!/bin/bash
mkdir build
cd build

export CC="gcc-5"
export CXX="g++-5"
cmake .. \
-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")  \
-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var(LIBDIR))") \
-DPYTHON_EXECUTABLE=$(which python)

cd build
make -j4

echo "add `pwd` to your PYTHONPATH"
