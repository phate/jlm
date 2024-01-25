#!/bin/bash
set -eu

# Default values for all tunables.
TARGET="release"
ENABLE_ASSERTS="no"
LLVM_CONFIG_BIN="llvm-config-16"
ENABLE_COVERAGE="no"
ENABLE_HLS=
CIRCT_PATH=

function usage()
{
	echo "Usage: ./configure.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "The following options can be set, with defaults specified in brackets:"
	echo "  --target MODE         Sets the build mode. Supported build modes are"
	echo "                        'debug' and 'release'. [${TARGET}]"
	echo "  --enable-asserts      Enables asserts."
	echo "  --enable-hls PATH     Enable the HLS backend, and sets the path to"
	echo "                        CIRCT, which the backend depends on."
	echo "  --llvm-config PATH    The llvm-config script used to determine up llvm"
	echo "                        build dependencies. [${LLVM_CONFIG_BIN}]"
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
			CIRCT_PATH="$1"
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
		--enable-coverage)
			ENABLE_COVERAGE="yes"
			shift
			;;
		--help)
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


CXXFLAGS_COMMON="--std=c++17 -Wall -Wpedantic -Wextra -Wno-unused-parameter -Werror -Wfatal-errors -gdwarf-4 -g"
CPPFLAGS_COMMON="-I. -Itests"

CPPFLAGS_LLVM=$(${LLVM_CONFIG_BIN} --cflags)

CXXFLAGS_TARGET=""
if [ "${TARGET}" == "release" ] ; then
	CXXFLAGS_TARGET="-O3"
elif [ "${TARGET}" == "debug" ] ; then
	CXXFLAGS_TARGET="-O0"
else
	echo "No build type set. Please select either 'debug' or 'release'." >&2
	exit 1
fi

CPPFLAGS_ASSERTS=""
if [ "${ENABLE_ASSERTS}" == "yes" ] ; then
	CPPFLAGS_ASSERTS="-DJLM_ENABLE_ASSERTS"
fi

CPPFLAGS_CIRCT=""
CXXFLAGS_CIRCT=""
if [ "${ENABLE_HLS}" == "yes" ] ; then
	CPPFLAGS_CIRCT="-I${CIRCT_PATH}/include"
	CXXFLAGS_CIRCT="-Wno-error=comment"
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
CXXFLAGS=${CXXFLAGS-} ${CXXFLAGS_COMMON} ${CXXFLAGS_TARGET} ${CXXFLAGS_CIRCT}
CPPFLAGS=${CPPFLAGS-} ${CPPFLAGS_COMMON} ${CPPFLAGS_LLVM} ${CPPFLAGS_ASSERTS} ${CPPFLAGS_CIRCT}
ENABLE_HLS=${ENABLE_HLS}
CIRCT_PATH=${CIRCT_PATH}
LLVMCONFIG=${LLVM_CONFIG_BIN}
ENABLE_COVERAGE=${ENABLE_COVERAGE}
EOF
	if [ ! -z "${CXX-}" ] ; then
		echo "CXX=${CXX}"
	fi
) > Makefile.config
