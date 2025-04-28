#!/bin/bash
set -eu

# URL to the benchmark git repository and the commit to be used
GIT_REPOSITORY=https://github.com/phate/polybench-jlm.git
GIT_COMMIT=2d784cc86cc8680e3a1fe6d79a54804e218fe1b9

# Get the absolute path to this script and set default JLM paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR=${SCRIPT_DIR}/..
JLM_BIN_DIR=${JLM_ROOT_DIR}/build

# Set default path for where the benchmark will be cloned and make target for running it
BENCHMARK_DIR=${JLM_ROOT_DIR}/usr/polybench
BENCHMARK_RUN_TARGET=check

# Use default compiler
JLC=""

# Execute benchmarks in parallel by default
if [[ "$OSTYPE" == "darwin"* ]]; then
  PARALLEL_THREADS=`sysctl -n hw.ncpu`
else
  PARALLEL_THREADS=`nproc`
fi

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./run-polybench.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --benchmark-path PATH The path where to place the polybench suite."
	echo "                        [${BENCHMARK_DIR}]"
	echo "  --parallel #THREADS   The number of threads to run in parallel."
	echo "                        Default=[${PARALLEL_THREADS}]"
	echo "  --mlir                Compile polybench with jlc-mlri-roundtrip.py."
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
		--parallel)
			shift
			PARALLEL_THREADS=$1
			shift
			;;
		--mlir)
			JLC="JLC=../../scripts/jlc-mlir-roundtrip.py"
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
else
	git -C ${BENCHMARK_DIR} fetch origin
fi

export PATH=${JLM_BIN_DIR}:${PATH}
cd ${BENCHMARK_DIR}
git checkout ${GIT_COMMIT}
make clean
echo "make ${JLC} -j ${PARALLEL_THREADS} -O ${BENCHMARK_RUN_TARGET}"
make ${JLC} -j ${PARALLEL_THREADS} -O ${BENCHMARK_RUN_TARGET}
