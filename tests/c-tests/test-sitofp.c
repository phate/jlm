#include <assert.h>

float
sitofp(int x)
{
	return x;
}

int
main()
{
	assert(sitofp(-4) == -4.0);

	return 0;
}
