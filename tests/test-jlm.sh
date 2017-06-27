#!	/bin/bash

if [ $# -eq 0 ] ; then
	echo "ERROR: No file supplied."
	exit 1
fi

root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

base=$(basename $1)
file="/tmp/${base%.*}"
clang_out="${file}.ll"
jlm_out="${file}-jlm.ll"
llc_out="${file}-jlm.o"
gcc_out="${file}-jlm"

clang-3.7 -O0 -S -emit-llvm $1 -o ${clang_out}
${root}/../jlm --j2l ${clang_out} 2> ${jlm_out}
llc-3.7 -O0 -filetype=obj -o ${llc_out} ${jlm_out}
clang-3.7 -O0 ${llc_out} -o ${gcc_out}
bash -c "${gcc_out}"
