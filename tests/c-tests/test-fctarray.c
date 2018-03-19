#include <assert.h>

unsigned int f()
{
	return 3;
}

unsigned int g()
{
	return 4;
}

unsigned int h()
{
	return 5;
}

int
main()
{
	unsigned int (*fctarray[3])() = {f, g, h};

	unsigned int sum = 0;
	for (unsigned int i = 0; i < 3; i++)
		sum += fctarray[i]();

	assert(sum == 12);

	return 0;
}
