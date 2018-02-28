#include <assert.h>

float
fptrunc(double d)
{
	return (float)d;
}

int
main()
{
	assert(fptrunc(0.4) == (float)0.4);

	return 0;
}
