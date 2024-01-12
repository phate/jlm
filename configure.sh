#!/bin/bash

# Default values for all tunables.
CIRCT_PATH=
TARGET="release"
LLVM_CONFIG_BIN="llvm-config-16"
ENABLE_COVERAGE="no"
CIRCT_ENABLED="no"

function usage()
{
	echo "Usage: ./configure [OPTION] [VAR=VALUE]"
	echo ""
	echo "The following options can be set, with defaults specified in brackets:"
	echo "  --target MODE         Sets the build mode. Supported build modes are"
	echo "                        'debug' and 'release'. [${TARGET}]"
	echo "  --circt-path PATH     Sets the path for the CIRCT tools and enables"
	echo "                        building with CIRCT support. [${CIRCT_PATH}]"
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
		--circt-path)
			shift
			CIRCT_PATH="$1"
			CIRCT_ENABLED="yes"
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


CXXFLAGS_COMMON="--std=c++17 -Wall -Wpedantic -Wextra -Wno-unused-parameter -Werror -Wfatal-errors -gdwarf-4 -g -fPIC"

CPPFLAGS_COMMON="-I. -Itests"

CPPFLAGS_LLVM=$(${LLVM_CONFIG_BIN} --cflags)

CPPFLAGS_CIRCT=""

if [ "${TARGET}" == "release" ] ; then
	CXXFLAGS_TARGET="-O3"
elif [ "${TARGET}" == "debug" ] ; then
	CXXFLAGS_TARGET="-O0"
	CPPFLAGS_TARGET="-DJLM_DEBUG -DJLM_ENABLE_ASSERTS"
else
	echo "No build type set. Please select either 'debug' or 'release'." >&2
	exit 1
fi

if [ "${CIRCT_ENABLED}" == "yes" ] ; then
	CPPFLAGS_CIRCT="-DCIRCT=1 -I${CIRCT_PATH}/include"
fi

CLANG_BIN=$(${LLVM_CONFIG_BIN} --bindir)

if [ "${ENABLE_COVERAGE}" == "yes" ] ; then
	if ! which gcovr >/dev/null ; then
		echo "Warning: gcovr is required for code coverage computation but could not be found on search path."
	fi
fi

mkdir -p build-"${TARGET}"
rm -rf build ; ln -sf build-"${TARGET}" build

(
	cat <<EOF
CXXFLAGS=${CXXFLAGS} ${CXXFLAGS_COMMON} ${CXXFLAGS_TARGET}
CPPFLAGS=${CPPFLAGS} ${CPPFLAGS_COMMON} ${CPPFLAGS_TARGET} ${CPPFLAGS_LLVM} ${CPPFLAGS_CIRCT}
CIRCT_PATH=${CIRCT_PATH}
LLVMCONFIG=${LLVM_CONFIG_BIN}
ENABLE_COVERAGE=${ENABLE_COVERAGE}
CLANG_BIN=${CLANG_BIN}
EOF
	if [ "${CXX}" != "" ] ; then
		echo "CXX=${CXX}"
	fi
) > Makefile.config
