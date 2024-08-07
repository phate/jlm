#!/bin/bash
set -eu

GIT_COMMIT=debf1ed774c2bbdbfc8e7bc987a21f72e8f08f65

# Get the absolute path to this script and set default build and install paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
CIRCT_BUILD=${JLM_ROOT_DIR}/build-circt
CIRCT_INSTALL=${JLM_ROOT_DIR}/usr
LLVM_LIT_PATH=/usr/local/bin/lit

LLVM_VERSION=17
LLVM_CONFIG_BIN=llvm-config-${LLVM_VERSION}

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./build-circt.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --llvm-config PATH    The llvm-config script used to determine up llvm"
	echo "                        build dependencies. [${LLVM_CONFIG_BIN}]"
	echo "  --build-path PATH     The path where to build CIRCT."
	echo "                        [${CIRCT_BUILD}]"
	echo "  --install-path PATH   The path where to install CIRCT."
	echo "                        [${CIRCT_INSTALL}]"
	echo "  --llvm-lit-path PATH  The path to the LLVM lit tool."
	echo "                        [${LLVM_LIT_PATH}]"
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--llvm-config)
			shift
			LLVM_CONFIG_BIN="$1"
			shift
			;;
		--build-path)
			shift
			CIRCT_BUILD=$(readlink -m "$1")
			shift
			;;
		--install-path)
			shift
			CIRCT_INSTALL=$(readlink -m "$1")
			shift
			;;
		--llvm-lit-path)
			shift
			LLVM_LIT_PATH=$(readlink -m "$1")
			shift
			;;
		--get-commit-hash)
			commit >&1
			exit 0
			;;
		--help)
			usage >&2
			exit 1
			;;
	esac
done

LLVM_BINDIR=$(${LLVM_CONFIG_BIN} --bindir)
LLVM_CMAKEDIR=$(${LLVM_CONFIG_BIN} --cmakedir)

CIRCT_GIT_DIR=${CIRCT_BUILD}/circt.git
CIRCT_BUILD_DIR=${CIRCT_BUILD}/build

if [ ! -d "$CIRCT_GIT_DIR" ] ;
then
	git clone https://github.com/EECS-NTNU/circt.git ${CIRCT_GIT_DIR}
fi
git -C ${CIRCT_GIT_DIR} checkout ${GIT_COMMIT}
cmake -G Ninja \
	${CIRCT_GIT_DIR} \
	-B ${CIRCT_BUILD_DIR} \
	-DCMAKE_C_COMPILER=${LLVM_BINDIR}/clang \
	-DCMAKE_CXX_COMPILER=${LLVM_BINDIR}/clang++ \
	-DCMAKE_BUILD_TYPE=RelWithDebInfo \
	-DLLVM_DIR=${LLVM_CMAKEDIR} \
	-DMLIR_DIR=${LLVM_CMAKEDIR}/../mlir \
	-DLLVM_EXTERNAL_LIT="${LLVM_LIT_PATH}" \
	-DLLVM_LIT_ARGS="-v --show-unsupported" \
	-DVERILATOR_DISABLE=ON \
	-DCMAKE_INSTALL_PREFIX=${CIRCT_INSTALL}
ninja -C ${CIRCT_BUILD_DIR}
ninja -C ${CIRCT_BUILD_DIR} install
