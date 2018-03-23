#include <assert.h>

double
ui2fp(unsigned i)
{
	return (double)(i);
}

int
main()
{
	assert(ui2fp(3) == 3.0);

	return 0;
}
