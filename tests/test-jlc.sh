#!/bin/bash

if [ $# -eq 0 ] ; then
	echo "ERROR: No file supplied."
	exit 1
fi

dir=$(dirname $2)

PATH=$PATH:$1/bin

mkdir -p $1/build/$dir
# jlc places the o-file in the same directory as the c-file
# So to place the o-file in build then we need to copy the c-file
cp $2 $1/build/$2
jlc -Wall -Werror -O3 -o $1/build/$2-jlm $1/build/$2 || exit 1
bash -c "$1/build/$2-jlm"
