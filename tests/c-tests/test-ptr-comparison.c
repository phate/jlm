#include <assert.h>

unsigned int
test_eq(unsigned int * x, unsigned int * y)
{
	return x == y;
}

unsigned int
test_ne(unsigned int * x, unsigned int * y)
{
	return x != y;
}

unsigned int
test_gt(unsigned int * x, unsigned int * y)
{
	return x > y;
}

unsigned int
test_ge(unsigned int * x, unsigned int * y)
{
	return x >= y;
}

unsigned int
test_lt(unsigned int * x, unsigned int * y)
{
	return x < y;
}

unsigned int
test_le(unsigned int * x, unsigned int * y)
{
	return x <= y;
}

int
main()
{
	unsigned int x = 4;
	unsigned int y = 5;

	assert(test_eq(&x, &y) == 0);
	assert(test_eq(&x, &x));

	assert(test_ne(&x, &y));
	assert(test_ne(&x, &x) == 0);

	assert(test_gt(&x, &x) == 0);

	assert(test_ge(&x, &x));

	assert(test_lt(&x, &x) == 0);

	assert(test_le(&x, &x));
}
