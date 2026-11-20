// RUN: %jlc -Wall -Werror -O3 %s -o %t && %t

#include <assert.h>

int
main(int argc, char ** argv)
{
	assert(argc == 1);
	return 0;
}
