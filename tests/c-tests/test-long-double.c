#include <assert.h>

long double
f()
{
	return 7.0;
}

int
main()
{
	assert(f() == 7.0);

	return 0;
}
