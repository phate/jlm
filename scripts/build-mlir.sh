#!/bin/bash
set -eu

GIT_COMMIT=6bfb270607f35b787bc849182f184e83548aa404

MLIR_BUILD="${PWD}/build-mlir/"
MLIR_INSTALL="${PWD}/usr/"

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./build-mlir.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --build-path PATH     The path where to build MLIR."
	echo "                        [${MLIR_BUILD}]"
	echo "  --install-path PATH   The path where to install MLIR."
	echo "                        [${MLIR_INSTALL}]"
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--build-path)
			shift
			MLIR_BUILD="$1"
			shift
			;;
		--install-path)
			shift
			MLIR_INSTALL="$1"
			shift
			;;
		--get-commit-hash)
			commit >&2
			exit 1
			;;
		--help)
			usage >&2
			exit 1
			;;
	esac
done

git clone https://github.com/EECS-NTNU/mlir_rvsdg.git ${MLIR_BUILD}
cd ${MLIR_BUILD}
git checkout ${GIT_COMMIT}
mkdir -p build
cd build
cmake -G Ninja .. \
        -DCMAKE_C_COMPILER=clang-16 \
        -DCMAKE_CXX_COMPILER=clang++-16 \
        -DLLVM_DIR=/usr/lib/llvm-16/cmake/ \
        -DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir \
        -DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL} \
	-Wno-dev
cmake --build .
ninja install
