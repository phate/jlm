#include <assert.h>

unsigned int
f()
{
	return 4;
}

unsigned int
g()
{
	return 3;
}

unsigned int
sum(unsigned int(*x)(), unsigned int(*y)())
{
	return x() + y();
}

int
main()
{
	assert(sum(f, g) == 7);
}
