#!/bin/bash
set -eu

GIT_COMMIT=093cdfe482530623fea01e1d3242af93e533ba54

# Get the absolute path to this script and set default build and install paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR=${SCRIPT_DIR}/..
CIRCT_BUILD=${JLM_ROOT_DIR}/build-circt
CIRCT_INSTALL=${JLM_ROOT_DIR}/usr

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./build-circt.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --build-path PATH     The path where to build CIRCT."
	echo "                        [${CIRCT_BUILD}]"
	echo "  --install-path PATH   The path where to install CIRCT."
	echo "                        [${CIRCT_INSTALL}]"
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--build-path)
			shift
			CIRCT_BUILD="$1"
			shift
			;;
		--install-path)
			shift
			CIRCT_INSTALL="$1"
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

CIRCT_GIT_DIR=${CIRCT_BUILD}/circt.git
CIRCT_BUILD_DIR=${CIRCT_BUILD}/build

if [ ! -d "$CIRCT_GIT_DIR" ] ;
then
	git clone https://github.com/EECS-NTNU/circt.git ${CIRCT_GIT_DIR}
fi
cd ${CIRCT_GIT_DIR}
git checkout ${GIT_COMMIT}
cmake -G Ninja \
	${CIRCT_GIT_DIR} \
	-B ${CIRCT_BUILD_DIR} \
	-DCMAKE_C_COMPILER=clang-16 \
	-DCMAKE_CXX_COMPILER=clang++-16 \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DLLVM_DIR=/usr/lib/llvm-16/cmake/ \
	-DMLIR_DIR=/usr/lib/llvm-16/lib/cmake/mlir \
	-DLLVM_EXTERNAL_LIT=/usr/local/bin/lit \
	-DLLVM_LIT_ARGS="-v --show-unsupported" \
	-DVERILATOR_DISABLE=ON \
	-DCMAKE_INSTALL_PREFIX=${CIRCT_INSTALL}
ninja -C ${CIRCT_BUILD_DIR}
# ninja -C ${CIRCT_BUILD_DIR} check-circt
ninja -C ${CIRCT_BUILD_DIR} install
