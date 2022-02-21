#include <stdio.h>

#define n 3

int
main()
{
	static const char * strings[n] = {"1", "2", "3"};

	for (unsigned int i = 0; i < n; i++)
		printf("%s", strings[i]);

	return 0;
}
