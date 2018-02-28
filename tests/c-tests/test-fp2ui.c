#include <assert.h>

unsigned int
fp2ui(double d)
{
	return (unsigned int)d;
}

int
main()
{
	assert(fp2ui(3.4) == 3);

	return 0;
}
