// RUN: %jlc -Wall -Werror -O3 %s -o %t && %t

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
