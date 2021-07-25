#include <assert.h>

typedef struct {
	unsigned int * addr;
	unsigned int type;
} object;

object
addr(object bar)
{
	return bar;
}

int
main()
{
	unsigned int nil = 0;
	object x = {&nil, 42};

	x = addr(x);
	assert(x.addr == &nil && x.type == 42);

	return 0;
}
