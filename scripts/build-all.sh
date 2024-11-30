#!/bin/bash
set -eu

LLVM_VERSION=18

# Get the absolute path to this script and set root path
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"

# Set operating system specific configurations
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  LLVM_CONFIG_BIN="llvm-config-"${LLVM_VERSION}
  MAKE_OPT="-j $(nproc) -O"
elif [[ "$OSTYPE" == "darwin"* ]]; then
  LLVM_CONFIG_BIN="/usr/local/Cellar/llvm@"${LLVM_VERSION}"/"${LLVM_VERSION}"*/bin/llvm-config"
  # Use the same MACOSX_DEPLOYMENT_TARGET as used for the LLVM and MLIR libraries
  # to aviod warnings during linking
  export MACOSX_DEPLOYMENT_TARGET="$(otool -l ${LLVM_CONFIG_BIN} | grep minos | awk '{print $2}')"
  MAKE_OPT="-j $(sysctl -n hw.ncpu)"
else
  echo "warning: Operating system not recognized." >&2
  LLVM_CONFIG_BIN=""
  MAKE_OPT=""
fi

# Check if llvm-config exists
LLVM_CONFIG_CHECK=`command -v ${LLVM_CONFIG_BIN} || true`
if [ -z "$LLVM_CONFIG_CHECK" ]; then
  echo "error: llvm-config could not be found." >&2
  echo "       Set the path to llvm-config manually." >&2
  exit 1
fi

# Check that lit is installed
LLVM_LIT_PATH=`command -v lit || true`
if [ -z "$LLVM_LIT_PATH" ]; then
  echo "error: lit could not be found in your PATH." >&2
  echo "       Install lit and make sure it is in your PATH." >&2
  exit 1
fi

# Default build options
BUILD_HLS="--enable-hls"
BUILD_MLIR="--enable-mlir"

function usage()
{
        echo "Usage: ./build-all.sh [OPTION]"
        echo ""
        echo "  --llvm-config PATH    The llvm-config script used to determine up llvm"
        echo "                        build dependencies. [${LLVM_CONFIG_BIN}]"
        echo "  --disable-mlir        Do not build the RVSDG MLIR Dialect and disable"
        echo "                        the MLIR backend/frontend in jlm."
        echo "  --disable-HLS         Do not build CIRCT and disable the HLS backend."
        echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
        case "$1" in
                --llvm-config)
                        shift
                        LLVM_CONFIG_BIN="$1"
                        shift
                        ;;
                --disable-mlir)
                        shift
                        BUILD_MLIR=""
                        shift
                        ;;
                --disable-hls)
                        shift
                        BUILD_HLS=""
                        shift
                        ;;
                --help)
                        usage >&2
                        exit 1
                        ;;
        esac
done

# Check that we are to build CIRCT and that it hasn't already been built
if [ -n ${BUILD_HLS} ] && [ ! -f ${JLM_ROOT_DIR}/usr/lib/libCIRCTFIRRTL.a ]; then
  echo "Building CIRCT"
  ${SCRIPT_DIR}/build-circt.sh --llvm-config ${LLVM_CONFIG_BIN}
fi

# Check that we are to build MLIR and that it hasn't already been built
if [ -n ${BUILD_MLIR} ] && [ ! -f ${JLM_ROOT_DIR}/usr/lib/libMLIRRVSDG.a ]; then
  echo "Building MLIR Dialect"
  ${SCRIPT_DIR}/build-mlir.sh --llvm-config ${LLVM_CONFIG_BIN}
fi

echo "Configuring jlm"
cd ${JLM_ROOT_DIR} && ./configure.sh ${BUILD_HLS} ${BUILD_MLIR} --llvm-config ${LLVM_CONFIG_BIN}

echo "Building jlm"
make -C ${JLM_ROOT_DIR} ${MAKE_OPT}

echo "Checking build of jlm"
export C_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/
make -C ${JLM_ROOT_DIR} ${MAKE_OPT} check
