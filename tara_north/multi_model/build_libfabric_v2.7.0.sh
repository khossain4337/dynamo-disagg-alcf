#!/bin/bash
# Upstream libfabric v2.7.0 (cxi + CUDA) with its json-c and curl deps.
# Compute node inside a PBS job (cxi devel headers exist only there), no args.
source ~/proxies.sh
module load gcc-native
# The nvhpc module exports CC=nvc; cmake and configure both honour it.
export CC=$(which gcc) CXX=$(which g++)
set -e
R=/vast/draco/tara/projects/Tara_Deployment/software/repositories/libfabric_v2.7.0
DEPS=$R/deps
PREFIX=$R/v2.7.0_cu13.0
CUDA=/opt/nvidia/hpc_sdk/Linux_aarch64/25.9/cuda/13.0
mkdir -p $R/src
exec > >(tee $R/build.log) 2>&1
cd $R/src

curl -LO https://github.com/ofiwg/libfabric/releases/download/v2.7.0/libfabric-2.7.0.tar.bz2
curl -LO https://curl.se/download/curl-8.14.1.tar.xz
curl -LO https://github.com/Kitware/CMake/releases/download/v3.31.7/cmake-3.31.7-linux-aarch64.tar.gz
git clone --depth 1 -b json-c-0.16-20220414 https://github.com/json-c/json-c.git
sha256sum libfabric-2.7.0.tar.bz2 curl-8.14.1.tar.xz cmake-3.31.7-linux-aarch64.tar.gz | tee SHA256SUMS
tar xf libfabric-2.7.0.tar.bz2
tar xf curl-8.14.1.tar.xz
tar xf cmake-3.31.7-linux-aarch64.tar.gz

# No system cmake on Tara; json-c builds only with CMake.
CMAKE=$R/src/cmake-3.31.7-linux-aarch64/bin/cmake
$CMAKE -S json-c -B json-c/build -DCMAKE_INSTALL_PREFIX=$DEPS -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_STATIC_LIBS=OFF
$CMAKE --build json-c/build -j 32
$CMAKE --install json-c/build

cd $R/src/curl-8.14.1
./configure --prefix=$DEPS --without-ssl --without-libpsl --disable-ldap
make -j 32 install

# rpath: libfabric finds json-c and curl without $DEPS on LD_LIBRARY_PATH.
cd $R/src/libfabric-2.7.0
./configure --prefix=$PREFIX \
  --disable-sockets --disable-udp --disable-verbs --disable-rxm --disable-mrail \
  --disable-rxd --disable-shm --disable-tcp --disable-usnic --disable-efa \
  --disable-psm2 --disable-psm3 --disable-opx \
  --enable-cxi --with-json-c=$DEPS --with-curl=$DEPS \
  --enable-cuda-dlopen --with-cuda=$CUDA \
  LDFLAGS="-Wl,-rpath,$DEPS/lib"
make -j 32 install

set +e
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
$PREFIX/bin/fi_info --version
ldd $PREFIX/lib/libfabric.so.1 | grep -E 'cxi|json|curl|not found'
mpiexec -n 1 --single-node-vni $PREFIX/bin/fi_info -p cxi | grep -E 'provider|domain'
$PREFIX/bin/fi_info -e -p cxi | grep -oE 'FI_CXI_(DEFAULT_CQ_SIZE|DEFAULT_TX_SIZE|DISABLE_HOST_REGISTER|RDZV_EAGER_SIZE|RDZV_GET_MIN|RDZV_PROTO|RDZV_THRESHOLD|ENABLE_WRITEDATA)' | sort -u
FI_CXI_ENABLE_WRITEDATA=1 mpiexec -n 1 --single-node-vni $PREFIX/bin/fi_info -p cxi -v | grep -m2 cq_data_size
mpiexec -n 1 --single-node-vni $PREFIX/bin/fi_info -p cxi -v | grep -m2 cq_data_size
