#include <assert.h>

int
fp2si(double d)
{
	return (int)d;
}

int
main()
{
	assert(fp2si(-3.4) == -3);

	return 0;
}
