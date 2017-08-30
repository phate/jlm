#include <stdio.h>

static void
f(size_t n)
{
	do {
		printf("foobar\n");
	} while (n--);
}

int
main()
{
	printf("%d + %d = %d\n", 3, 4, 7);
	f(5);
	return 0;
}
