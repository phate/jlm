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
