#include <assert.h>
#include <stdarg.h>

int
computeSum(int x, ...)
{
	va_list ap;

	va_start(ap, x);
	int y = va_arg(ap, int);
	int z = va_arg(ap, int);
	va_end(ap);

	return x+y+z;
}

int
f3(int i, ...)
{
	va_list aps[10];
	va_start(aps[4], i);
	int x = va_arg(aps[4], int);
	va_end(aps[4]);

	return x;
}

int
main()
{
	int sum = computeSum(42, 3, 5);
	assert(sum == 50);

	int x = f3(4, 3);
	assert(x == 3);

	return 0;
}
