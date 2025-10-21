#!/bin/bash
set -eu

# URL to the SPEC git repository and the commit to be used
GIT_REPOSITORY=git@github.com:phate/spec-cpu2017-jlm.git
GIT_COMMIT=534583305d8644fd387877684732f2f6001ff9da

# Get the absolute path to this script and set default JLM paths
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
JLM_ROOT_DIR="$(realpath "${SCRIPT_DIR}/..")"
JLM_BIN_DIR=${JLM_ROOT_DIR}/build

# Set default path for where the benchmark will be cloned and make target for running it
SPEC_DIR=${JLM_ROOT_DIR}/usr/spec-cpu2017
SPEC_RUN_TARGET=cpu2017-run
TAR_FILE=""

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
	echo "Usage: ./run-spec.sh [OPTION] [VAR=VALUE]"
	echo ""
	echo "  --spec-path PATH      The path where to place the SPEC benchmarks."
	echo "                        Default=[${SPEC_DIR}]"
	echo "  --tar-file PATH       The tar-file containing the SPEC benchmarks."
	echo "                        An absolute path is required."
 	echo "  --parallel #THREADS   The number of threads to run in parallel."
	echo "                        Default=[${PARALLEL_THREADS}]"
	echo "  --get-commit-hash     Prints the commit hash used for the build."
	echo "  --help                Prints this message and stops."
}

while [[ "$#" -ge 1 ]] ; do
	case "$1" in
		--spec-path)
			shift
			SPEC_DIR=$(readlink -m "$1")
			shift
			;;
		--tar-file)
			shift
			TAR_FILE=$(readlink -m "$1")
			shift
			;;
		--parallel)
			shift
			PARALLEL_THREADS=$1
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

if [ ! -d "$SPEC_DIR" ] ;
then
	git clone ${GIT_REPOSITORY} ${SPEC_DIR}
else
	git -C ${SPEC_DIR} fetch origin
fi

export PATH=${JLM_BIN_DIR}:${PATH}
cd ${SPEC_DIR}
git checkout ${GIT_COMMIT}

if [ ! -f "$SPEC_DIR/.installed" ] ;
then
if [ ! -f "$TAR_FILE" ] ;
then
	echo "You need to provide an absolute path to the TAR-file that contains SPEC:"
	echo "    ./run-spec.sh --tar-file [absolute-path-to-file]"
	exit
fi
	echo "make install FILE=${TAR_FILE}"
	make install FILE=${TAR_FILE}
fi

echo "make NCPU=${PARALLEL_THREADS} ${SPEC_RUN_TARGET}"
make NCPU=${PARALLEL_THREADS} ${SPEC_RUN_TARGET}
