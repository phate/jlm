// RUN: %jlc -Wall -Werror -O3 %s -o %t && %t

#include <stdio.h>

int
main()
{
	fprintf(stderr, "%d * %d = %d", 3, 3, 9);
}
