#!/bin/bash
set -eu +x

# URL to the benchmark git repository and the commit to be used
GIT_REPOSITORY=https://github.com/haved/jlm-benchmark.git
GIT_COMMIT=5e0451d4ba7282b2a152787aef98de09403bf4cb

# Get the absolute path to this script and set default JLM paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"

# Set default path for where the benchmark will be cloned and make target for running it
BENCHMARK_DIR=${JLM_ROOT_DIR}/usr/benchmarks

# Execute benchmarks in parallel by default
if [[ "$OSTYPE" == "darwin"* ]]; then
	PARALLEL_THREADS=`sysctl -n hw.ncpu`
else
	PARALLEL_THREADS=`nproc`
fi

APT_INSTALL_DEPS=false
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
	echo "  --apt-install-deps    For CI runner or Ubuntu 24. Installs apt package dependencies before running."
 	echo "  --parallel #THREADS   The number of threads to run in parallel."
	echo "                        Default=[${PARALLEL_THREADS}]"
	echo "  --benchmark BENCH     Only extract and build a specific benchamrk."
	echo "                        Default=[ALL]"
	echo "                        BENCH=[polybench|spec|emacs|ghostscript|gdb|sendmail]"
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
			BENCHMARK_DIR="$(readlink -m "$1")"
			shift
			;;
		--apt-install-deps)
			APT_INSTALL_DEPS=true
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
	./apt-install-dependencies.sh
fi

if [ "${CLEAN}" = true ]; then
	./run.sh --clean
	exit  0
fi

./run.sh --jlm-opt "${JLM_ROOT_DIR}/build/jlm-opt" --llvm-config "${LLVMCONFIG}" --parallel "${PARALLEL_THREADS}" ${BENCHMARK}
