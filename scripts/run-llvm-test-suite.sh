#!/bin/bash
set -eu

# URL to the benchmark git repository and the commit to be used
GIT_REPOSITORY=https://github.com/phate/llvm-test-suite.git
GIT_COMMIT=ebdef97621d4e024dca3ec0095de958e6ccb3ad8

# Get the absolute path to this script and set default JLM paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR=${SCRIPT_DIR}/..
JLM_BIN_DIR=${JLM_ROOT_DIR}/build

# Set default path for where the benchmark will be cloned and make target for running it
BENCHMARK_DIR=${JLM_ROOT_DIR}/usr/llvm-test-suite
BENCHMARK_RUN_TARGET=llvm-run-opt

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./run-llvm-test.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --benchmark-path PATH The path where to place the llvm test suite."
	echo "                        [${BENCHMARK_DIR}]"
	echo "  --make-target TARGET  The make target to run."
	echo "                        [${BENCHMARK_RUN_TARGET}]"
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--benchmark-path)
			shift
			BENCHMARK_DIR=$(readlink -m "$1")
			shift
			;;
		--make-target)
			shift
			BENCHMARK_RUN_TARGET=$1
			shift
			;;
		--get-commit-hash)
			commit >&2
			exit 1
			;;
		--help|*)
			usage >&2
			exit 1
			;;
	esac
done

if [ ! -d "$BENCHMARK_DIR" ] ;
then
	git clone ${GIT_REPOSITORY} ${BENCHMARK_DIR}
fi

export PATH=${JLM_BIN_DIR}:${PATH}
cd ${BENCHMARK_DIR}
git checkout ${GIT_COMMIT}
cd jlm
make clean
make ${BENCHMARK_RUN_TARGET}
