#include <stdio.h>

void
f(unsigned n)
{
	printf("%d\n", n);
	if (n > 0)
		f(n-1);
}

unsigned y(unsigned);

unsigned
x(unsigned n)
{
	return y(n);
}

unsigned
y(unsigned n)
{
	return x(n);
}

int
main()
{
	f(10);
	return 0;
}
