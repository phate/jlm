#!/bin/bash

function headers_from_deps() {
	cat $* | \
	sed -E -e "s/ /\\n/g" | \
	sed -E \
		-e "/^.*:/d" \
		-e "s/\\\\\$//" \
		-e "/^jlm/p" \
		-e "/^test/p" \
		-e "d" | \
	sort -u
}

declare DEPFILES=()
while [[ "$#" -ge 1 ]] ; do
	if [ "$1" == "-h" ] ; then
		shift
		break
	fi
	DEPFILES+=("$1")
	shift
done

declare HEADERS=()
while [[ "$#" -ge 1 ]] ; do
	if [ "$1" == "-s" ] ; then
		shift
		break
	fi
	HEADERS+=("$1")
	shift
done

declare SOURCES=()
while [[ "$#" -ge 1 ]] ; do
	SOURCES+=("$1")
	shift
done

TMPDIR=`mktemp -d`
trap 'rm -rf "${TMPDIR}"' EXIT

headers_from_deps "${DEPFILES[*]}" > "${TMPDIR}/headers_used"
(IFS='
' ; echo "${HEADERS[*]}" ; echo "${SOURCES[*]}" ) | sort -u > "${TMPDIR}/headers_declared"

if grep -f "${TMPDIR}/headers_declared" -v "${TMPDIR}/headers_used" > "${TMPDIR}/headers_undeclared" ; then
	echo "*** The following headers are used but not declared in build rules: ***"
	cat "${TMPDIR}/headers_undeclared"
	echo "Hint: the list may be inaccurate if dependence information is stale".
	echo "If you think this is the case, please try running 'make depclean ; make depend'."
	exit 1
else
	exit 0
fi
