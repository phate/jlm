#!/bin/bash

GIT_COMMIT=093cdfe482530623fea01e1d3242af93e533ba54

CIRCT_BUILD=./build-circt
CIRCT_INSTALL=./../circt

function commit()
{
	echo ${GIT_COMMIT}
}

while [[ "$#" -ge 1 ]] ; do
        case "$1" in
                --get-commit-hash)
                        commit >&2
                        exit 1
                        ;;
        esac
done

git clone https://github.com/EECS-NTNU/circt.git ${CIRCT_BUILD}
cd ${CIRCT_BUILD}
git checkout ${GIT_COMMIT}
mkdir build
cd build
cmake -G Ninja .. \
	-DCMAKE_C_COMPILER=clang-16 \
	-DCMAKE_CXX_COMPILER=clang++-16 \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DLLVM_DIR=/usr/lib/llvm-16/cmake/ \
	-DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir \
	-DLLVM_EXTERNAL_LIT=/usr/local/bin/lit \
	-DLLVM_LIT_ARGS="-v --show-unsupported" \
	-DVERILATOR_DISABLE=ON \
	-DCMAKE_INSTALL_PREFIX=$CIRCT_INSTALL
ninja
ninja check-circt
ninja install
