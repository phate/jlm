#!/bin/bash
set -eu

GIT_REPOSITORY=https://github.com/EECS-NTNU/mlir_rvsdg.git
GIT_COMMIT=e01d4ef44766b2da278e5a4e48d80d877c8017ce

# Get the absolute path to this script and set default build and install paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
MLIR_BUILD=${JLM_ROOT_DIR}/build-mlir
MLIR_INSTALL=${JLM_ROOT_DIR}/usr

LLVM_VERSION=18
LLVM_CONFIG_BIN=llvm-config-${LLVM_VERSION}

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./build-mlir.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --llvm-config PATH    The llvm-config script used to determine up llvm"
	echo "                        build dependencies. [${LLVM_CONFIG_BIN}]"
	echo "  --build-path PATH     The path where to build MLIR."
	echo "                        [${MLIR_BUILD}]"
	echo "  --install-path PATH   The path where to install MLIR."
	echo "                        [${MLIR_INSTALL}]"
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
			MLIR_BUILD=$(readlink -m "$1")
			shift
			;;
		--install-path)
			shift
			MLIR_INSTALL=$(readlink -m "$1")
			shift
			;;
		--get-commit-hash)
			commit >&1
			exit 0
			;;
		--help|*)
			usage >&2
			exit 1
			;;
	esac
done

LLVM_BINDIR=$(${LLVM_CONFIG_BIN} --bindir)
LLVM_CMAKEDIR=$(${LLVM_CONFIG_BIN} --cmakedir)

MLIR_GIT_DIR=${MLIR_BUILD}/mlir_rvsdg.git
MLIR_BUILD_DIR=${MLIR_BUILD}/build

if [ ! -d "$MLIR_GIT_DIR" ] ;
then
	git clone ${GIT_REPOSITORY} ${MLIR_GIT_DIR}
fi

git -C ${MLIR_GIT_DIR} checkout ${GIT_COMMIT}
cmake -G Ninja \
	${MLIR_GIT_DIR} \
	-B ${MLIR_BUILD_DIR} \
	-DLLVM_DIR=${LLVM_CMAKEDIR} \
	-DMLIR_DIR=${LLVM_CMAKEDIR}/../mlir \
	-DCMAKE_PREFIX_PATH=${LLVM_CMAKEDIR}/../mlir \
	-DCMAKE_INSTALL_PREFIX=${MLIR_INSTALL} \
	-Wno-dev
ninja -C ${MLIR_BUILD_DIR}
ninja -C ${MLIR_BUILD_DIR} install
