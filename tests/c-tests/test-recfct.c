#include <stdio.h>

void
f(unsigned n)
{
	printf("%d\n", n);
	if (n > 0)
		f(n-1);
}

int
main()
{
	f(10);
	return 0;
}
