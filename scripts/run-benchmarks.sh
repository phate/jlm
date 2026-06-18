#!/bin/bash
set -eu +x

# URL to the benchmark git repository and the commit to be used
GIT_REPOSITORY=https://github.com/haved/jlm-benchmark.git
GIT_COMMIT=50bf014fe8dacfaad7bbbfdd620b5b839518153c

# Get the absolute path to this script and set default JLM paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"

# Set default path for where the benchmark will be cloned and make target for running it
BENCHMARK_DIR=${JLM_ROOT_DIR}/usr/benchmarks

# Execute benchmarks in parallel by default
if [[ "$OSTYPE" == "darwin"* ]]; then
	NUM_PARALLEL_THREADS=`sysctl -n hw.ncpu`
else
	NUM_PARALLEL_THREADS=`nproc`
fi

APT_INSTALL_DEPS=false
CLEAN_RUNS=false
RUN_OPTIONS=""

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
	echo "  --apt-install-deps    For CI runner or Ubuntu 24. Installs apt package dependencies before running."
	echo "  --ci                  Use a benchmark configuration intended for running and validating in CI."
 	echo "  --parallel #THREADS   The number of threads to run in parallel."
	echo "                        Default=[${NUM_PARALLEL_THREADS}]"
	echo "  --benchmark BENCH     Only extract and build a specific benchamrk."
	echo "                        Default=[ALL]. See list of options in benchmark's run.sh"
	echo "  --clean-runs          Delete build files and statistics from previous runs."
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--clean-runs)
			CLEAN_RUNS=true
			shift
			;;
		--path)
			shift
			BENCHMARK_DIR="$(readlink -m "$1")"
			shift
			;;
		--apt-install-deps)
			APT_INSTALL_DEPS=true
			shift
			;;
		--ci)
			RUN_OPTIONS="${RUN_OPTIONS} --ci"
			shift
			;;
		--parallel)
			shift
			NUM_PARALLEL_THREADS=$1
			shift
			;;
		--benchmark)
			shift
			RUN_OPTIONS="${RUN_OPTIONS} --$1"
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

# Extract LLVMCONFIG from the Makefile.config used to build jlm-opt
source <(grep LLVMCONFIG= "${JLM_ROOT_DIR}/Makefile.config" || true)
if [[ -z "${LLVMCONFIG:-}" ]]; then
	echo "Unable to extract LLVMCONFIG path from Makefile.config"
	exit 1
fi

# Clone the benchmark repository
if [ ! -d "${BENCHMARK_DIR}" ] ;
then
	git clone ${GIT_REPOSITORY} "${BENCHMARK_DIR}"
else
	git -C "${BENCHMARK_DIR}" fetch origin
fi

cd "${BENCHMARK_DIR}"
git checkout ${GIT_COMMIT}

if [[ "${APT_INSTALL_DEPS}" = true ]]; then
	sudo ./apt-install-dependencies.sh
fi

if [ "${CLEAN_RUNS}" = true ]; then
	./run.sh clean-runs
	exit 0
fi

./run.sh --jlm-opt "${JLM_ROOT_DIR}/build/jlm-opt" --llvm-config "${LLVMCONFIG}" --parallel "${NUM_PARALLEL_THREADS}" ${RUN_OPTIONS}
