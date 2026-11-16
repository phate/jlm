// RUN: %jlc -Wall -Werror -O3 %s -o %t && %t

#include <assert.h>
#include <stdlib.h>

struct type {
	unsigned int * p;
	unsigned int x;
} t;

int
main()
{
	assert(t.p == NULL);
	assert(t.x == 0);
}
