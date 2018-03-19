#include <assert.h>

unsigned int
fib(unsigned int x)
{
	if (x <= 2)
		return 1;

	return fib(x-1) + fib(x-2);
}

int
main()
{
	assert(fib(7) == 13);
}
