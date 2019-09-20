#include <assert.h>

unsigned int
value(unsigned int i, unsigned int j, unsigned int k)
{
	return i == 42 ? j : k;
}


int
main()
{
	assert(value(42, 3, 4) == 3);
	assert(value(41, 3, 4) == 4);
}
