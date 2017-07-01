#include <assert.h>
#include <stdlib.h>

size_t
fac(size_t n)
{
	size_t r = 1;
	while (n > 1) {
		r = r*n;
		n--;
	}

	return r;
}

int
main()
{
	assert(fac(0) == 1);
	assert(fac(1) == 1);
	assert(fac(2) == 2);
	assert(fac(7) == 5040);
	return 0;
}
