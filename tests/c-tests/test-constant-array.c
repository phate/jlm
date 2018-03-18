#include <stdio.h>

int
main()
{
	const unsigned int n = 3;
	static const char * strings[n] = {"1", "2", "3"};

	for (unsigned int i = 0; i < n; i++)
		printf("%s", strings[i]);

	return 0;
}
