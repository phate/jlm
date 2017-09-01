#include <assert.h>
#include <stdio.h>

static unsigned int
f(unsigned int n)
{
	unsigned int s = 0;
	do {
		s += n;
	} while(n--);

	return s;
}

int
main()
{
	unsigned int s = f(5);
	printf("%d\n", s);
	assert(s == 15);
}
