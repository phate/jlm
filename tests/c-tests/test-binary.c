#include <assert.h>
#include <inttypes.h>
#include <stdlib.h>

typedef struct foo {
	int32_t i;
	int32_t k;
} foo;

static int
f(int k)
{
	return k;
}

int
main()
{
	/* offsetof from stddef.h */
	int v = f(((size_t)(&(((foo *)0)->k)))+45);
	assert(v == 49);

	return 0;
}
