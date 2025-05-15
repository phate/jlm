#!/bin/bash
set -eu

LLVM_VERSION=18

# Default values for all tunables.
TARGET="release"
ENABLE_ASSERTS="no"
LLVM_CONFIG_BIN="llvm-config-"${LLVM_VERSION}
ENABLE_COVERAGE="no"
ENABLE_HLS=
CIRCT_PATH="./usr"
CIRCT_LDFLAGS=
ENABLE_MLIR=
MLIR_PATH="./usr"
MLIR_LDFLAGS=

function usage()
{
	echo "Usage: ./configure.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "The following options can be set, with defaults specified in brackets:"
	echo "  --target MODE         Sets the build mode. Supported build modes are"
	echo "                        'debug' and 'release'. [${TARGET}]"
	echo "  --enable-asserts      Enables asserts."
	echo "  --enable-hls[=PATH]   Enable the HLS backend, and optionally sets a custom"
	echo "                        path to the CIRCT installation. [${CIRCT_PATH}]"
	echo "  --llvm-config PATH    The llvm-config script used to determine up llvm"
	echo "                        build dependencies. [${LLVM_CONFIG_BIN}]"
	echo "  --enable-mlir[=PATH]  Enables the MLIR Dialect, and optionally sets a custom"
	echo "                        path to the MLIR dialect installation. [${MLIR_PATH}]"
	echo "  --enable-coverage     Enable test coverage computation target."
	echo "  --help                Prints this message and stops."
	echo
	echo "Influential variables that can be set:"
	echo "  CXX        C++ compiler command"
	echo "  CXXFLAGS   C++ compiler flags"
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--target)
			shift
			TARGET="$1"
			shift
			;;
		--enable-hls)
			ENABLE_HLS="yes"
			shift
			;;
		--enable-hls=*)
			ENABLE_HLS="yes"
			CIRCT_PATH="${1#*=}"
			shift
			;;
		--enable-asserts)
			ENABLE_ASSERTS="yes"
			shift
			;;
		--llvm-config)
			shift
			LLVM_CONFIG_BIN="$1"
			shift
			;;
		--enable-mlir)
			ENABLE_MLIR="yes"
			shift
			;;
		--enable-mlir=*)
			ENABLE_MLIR="yes"
			MLIR_PATH="${1#*=}"
			shift
			;;
		--enable-coverage)
			ENABLE_COVERAGE="yes"
			shift
			;;
		-*) # Any unknown option triggers the usage text. This includes --help and -h
			usage >&2
			exit 1
			;;
		*=*)
			VARNAME=${1%%=*}
			VARVAL=${1#*=}
			eval "${VARNAME}"="${VARVAL}"
			shift
			;;
		*)
			usage >&2
			exit 1
			;;
	esac
done


CXXFLAGS_COMMON="--std=c++17 -Wall -Wpedantic -Wextra -Wno-unused-parameter -gdwarf-4 -g"
CPPFLAGS_COMMON="-I. -Itests"

CPPFLAGS_LLVM=$(${LLVM_CONFIG_BIN} --cflags)

CXXFLAGS_TARGET=""
if [ "${TARGET}" == "release" ] ; then
	CXXFLAGS_TARGET="-O3"
elif [ "${TARGET}" == "debug" ] ; then
	CXXFLAGS_TARGET="-O0"
elif [ "${TARGET}" == "release-anf" ] ; then
	CXXFLAGS_TARGET="-O3 -DANDERSEN_NO_FLAGS"
elif [ "${TARGET}" == "debug-anf" ] ; then
	CXXFLAGS_TARGET="-O0 -DANDERSEN_NO_FLAGS"
else
	echo "No build type set. Please select either 'debug' or 'release'." >&2
	exit 1
fi

CPPFLAGS_ASSERTS=""
if [ "${ENABLE_ASSERTS}" == "yes" ] ; then
	CPPFLAGS_ASSERTS="-DJLM_ENABLE_ASSERTS"
fi

CPPFLAGS_CIRCT=""
CXXFLAGS_NO_COMMENT=""
if [ "${ENABLE_HLS}" == "yes" ] ; then
	CPPFLAGS_CIRCT="-I${CIRCT_PATH}/include"
	CXXFLAGS_NO_COMMENT="-Wno-error=comment"
	CIRCT_LDFLAGS_ARRAY=(
		"-L${CIRCT_PATH}/lib"
		"-lCIRCTAnalysisTestPasses"
		"-lCIRCTDependenceAnalysis"
		"-lCIRCTExportFIRRTL"
		"-lCIRCTScheduling"
		"-lCIRCTSchedulingAnalysis"
		"-lCIRCTFirtool"
		"-lCIRCTFIRRTLReductions"
		"-lCIRCTFIRRTLToHW"
		"-lCIRCTExportVerilog"
		"-lCIRCTImportFIRFile"
		"-lCIRCTFIRRTLTransforms"
		"-lCIRCTHWTransforms"
		"-lCIRCTOMTransforms"
		"-lCIRCTSim"
		"-lCIRCTSVTransforms"
		"-lCIRCTTransforms"
		"-lCIRCTTargetDebugInfo"
		"-lCIRCTSV"
		"-lCIRCTComb"
		"-lCIRCTSupport"
		"-lCIRCTDebug"
		"-lCIRCTLTL"
		"-lCIRCTVerif"
		"-lCIRCTFIRRTL"
		"-lCIRCTSeq"
		"-lCIRCTSeqToSV"
		"-lCIRCTSeqTransforms"
		"-lCIRCTHW"
		"-lCIRCTVerifToSV"
		"-lCIRCTSimToSV"
		"-lCIRCTExportChiselInterface"
		"-lCIRCTOM"
		"-lMLIR"
	)
fi

CPPFLAGS_MLIR=""
if [ "${ENABLE_MLIR}" == "yes" ] ; then
	CPPFLAGS_MLIR="-I${MLIR_PATH}/include -DENABLE_MLIR"
	CXXFLAGS_NO_COMMENT="-Wno-error=comment"
	MLIR_LDFLAGS="-L${MLIR_PATH}/lib -lMLIRJLM -lMLIRRVSDG -lMLIR"
fi

if [ "${ENABLE_COVERAGE}" == "yes" ] ; then
	if ! which gcovr >/dev/null ; then
		echo "Warning: gcovr is required for code coverage computation but could not be found on search path."
	fi
fi

mkdir -p build-"${TARGET}"
rm -rf build ; ln -sf build-"${TARGET}" build

(
	cat <<EOF
CXXFLAGS=${CXXFLAGS-} ${CXXFLAGS_COMMON} ${CXXFLAGS_TARGET} ${CXXFLAGS_NO_COMMENT}
CPPFLAGS=${CPPFLAGS-} ${CPPFLAGS_COMMON} ${CPPFLAGS_LLVM} ${CPPFLAGS_ASSERTS} ${CPPFLAGS_CIRCT} ${CPPFLAGS_MLIR}
ENABLE_HLS=${ENABLE_HLS}
CIRCT_PATH=${CIRCT_PATH}
CIRCT_LDFLAGS=${CIRCT_LDFLAGS_ARRAY[*]}
ENABLE_MLIR=${ENABLE_MLIR}
MLIR_PATH=${MLIR_PATH}
MLIR_LDFLAGS=${MLIR_LDFLAGS}
LLVMCONFIG=${LLVM_CONFIG_BIN}
LLVM_VERSION=${LLVM_VERSION}
ENABLE_COVERAGE=${ENABLE_COVERAGE}
export LD_LIBRARY_PATH=$(${LLVM_CONFIG_BIN} --libdir)
EOF
	if [ ! -z "${CXX-}" ] ; then
		echo "CXX=${CXX}"
	fi
) > Makefile.config
