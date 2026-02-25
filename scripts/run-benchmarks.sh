#!/bin/bash
set -eu

# URL to the benchmark git repository and the commit to be used
GIT_REPOSITORY=https://github.com/haved/jlm-benchmark.git
GIT_COMMIT=c12be582bcd1d3da1c6d82c26ac76cdfffc84086

# Get the absolute path to this script and set default JLM paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
JLM_BIN_DIR=${JLM_ROOT_DIR}/build

# Set default path for where the benchmark will be cloned and make target for running it
BENCHMARK_DIR=${JLM_ROOT_DIR}/usr/benchmarks

# Execute benchmarks in parallel by default
if [[ "$OSTYPE" == "darwin"* ]]; then
	PARALLEL_THREADS=`sysctl -n hw.ncpu`
else
	PARALLEL_THREADS=`nproc`
fi

CLEAN=false
BENCHMARK=""

function commit()
{
	echo ${GIT_COMMIT}
}

function usage()
{
	echo "Usage: ./run-spec.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --path PATH           The path where to place the benchmarks."
	echo "                        Default=[${BENCHMARK_DIR}]"
 	echo "  --parallel #THREADS   The number of threads to run in parallel."
	echo "                        Default=[${PARALLEL_THREADS}]"
	echo "  --benchmark BENCH     Only extract and build a specific benchamrk."
	echo "                        Default=[ALL]"
	echo "                        BENCH=[polybench|spec|emacs|ghostscript|gdb|sendmail"
	echo "  --clean               Delete extracted sources and build files."
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--clean)
			CLEAN=true
			shift
			;;
		--path)
			shift
			BENCHMARK_DIR=$(readlink -m "$1")
			shift
			;;
		--parallel)
			shift
			PARALLEL_THREADS=$1
			shift
			;;
		--benchmark)
			shift
			BENCHMARK="--$1"
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

if [ ${CLEAN} = true ]; then
	./run-ci.sh --clean
	exit 1
fi

CLANG=$(${JLM_ROOT_DIR}/build/jlc a.c "-###" | head -n1 | cut "-d " -f1)
LLVM_BIN="$(dirname "${CLANG}")"

echo "./run.sh --jlm-opt ${JLM_ROOT_DIR}/build/jlm-opt --llvm-bin ${LLVM_BIN} --parallel ${PARALLEL_THREADS} ${BENCHMARK}"
./run.sh --jlm-opt ${JLM_ROOT_DIR}/build/jlm-opt --llvm-bin ${LLVM_BIN} --parallel ${PARALLEL_THREADS} ${BENCHMARK}
