#include <stdio.h>

struct {
	char * inplace;
	char data[];
} s = {0, {" foobar "}};

int
main()
{
	printf("%ld, '%s'\n", (long) s.inplace, s.data);

	return 0;
}
