#!	/bin/bash

if [ $# -eq 0 ] ; then
	echo "ERROR: No file supplied."
	exit 1
fi

root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

base=$(basename $1)
file="/tmp/${base%.*}"

PATH=$PATH:${root}/..

jlc -Wall -Werror -O0 -o ${file}-jlm $1 || exit 1
bash -c "${file}-jlm"
