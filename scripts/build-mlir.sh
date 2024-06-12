#!/bin/bash
set -eu

GIT_COMMIT=ab630d5a881a0e8fc5bdfa63a5984186fa9096c0

# Get the absolute path to this script and set default build and install paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR=${SCRIPT_DIR}/..
MLIR_BUILD=${JLM_ROOT_DIR}/build-mlir
MLIR_INSTALL=${JLM_ROOT_DIR}/usr

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./build-circt.sh [OPTION] [VAR=VALUE]"
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
			MLIR_BUILD="${PWD}/$1"
			shift
			;;
		--install-path)
			shift
			MLIR_INSTALL="${PWD}/$1"
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

MLIR_GIT_DIR=${MLIR_BUILD}/mlir_rvsdg.git
MLIR_BUILD_DIR=${MLIR_BUILD}/build

if [ ! -d "$MLIR_GIT_DIR" ] ;
then
	git clone https://github.com/EECS-NTNU/mlir_rvsdg.git ${MLIR_GIT_DIR}
fi
cd ${MLIR_GIT_DIR}
git checkout ${GIT_COMMIT}
cmake -G Ninja \
	${MLIR_GIT_DIR} \
	-B ${MLIR_BUILD_DIR} \
	-DCMAKE_C_COMPILER=clang-17 \
	-DCMAKE_CXX_COMPILER=clang++-17 \
	-DLLVM_DIR=/usr/lib/llvm-17/cmake/ \
	-DMLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir \
	-DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL} \
	-Wno-dev
cmake --build ${MLIR_BUILD_DIR}
ninja -C ${MLIR_BUILD_DIR} install
