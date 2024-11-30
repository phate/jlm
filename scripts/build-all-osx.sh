#!/bin/bash
set -eu

# Get the absolute path to this script and set default build and install paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
LLVM_CONFIG_BIN="/usr/local/Cellar/llvm@18/18.1.8/bin/llvm-config"
LLVM_LIT_PATH=`command -v lit || true`
if [ -z "$LLVM_LIT_PATH" ]; then
  echo "error: lit could not be found in your PATH" >&2
  echo "       Install lit with, e.g., brew install lit" >&2
  exit 1
fi

function usage()
{
        echo "Usage: ./build-all-osx.sh [OPTION] [VAR=VALUE]"
        echo ""
        echo "  --llvm-config PATH    The llvm-config script used to determine up llvm"
        echo "                        build dependencies. [${LLVM_CONFIG_BIN}]"
        echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
        case "$1" in
                --llvm-config)
                        shift
                        LLVM_CONFIG_BIN="$1"
                        shift
                        ;;
                --help)
                        usage >&2
                        exit 1
                        ;;
        esac
done

# Use the same MACOSX_DEPLOYMENT_TARGET as used for the LLVM and MLIR libraries
# to aviod warnings during linking
export MACOSX_DEPLOYMENT_TARGET="$(otool -l ${LLVM_CONFIG_BIN} | grep minos | awk '{print $2}')"

if [ ! -f ${JLM_ROOT_DIR}/usr/lib/libCIRCTFIRRTL.a ]; then
  echo "Building CIRCT"
  ${SCRIPT_DIR}/build-circt.sh --llvm-config ${LLVM_CONFIG_BIN}
fi

if [ ! -f ${JLM_ROOT_DIR}/usr/lib/libMLIRRVSDG.a ]; then
  echo "Building MLIR Dialect"
  ${SCRIPT_DIR}/build-mlir.sh --llvm-config ${LLVM_CONFIG_BIN}
fi

if [ ! -f ${JLM_ROOT_DIR}/Makefile.config ]; then
  echo "Configuring jlm"
  cd ${JLM_ROOT_DIR} && ./configure.sh --enable-hls --enable-mlir --llvm-config ${LLVM_CONFIG_BIN}
fi

echo "Building jlm"
make -C ${JLM_ROOT_DIR} -j`sysctl -n hw.ncpu`

echo "Checking build of jlm"
export C_INCLUDE_PATH=/Library/Developer/CommandLineTools/SDKs/MacOSX11.0.sdk/usr/include/
make -C ${JLM_ROOT_DIR} -j`sysctl -n hw.ncpu` check
