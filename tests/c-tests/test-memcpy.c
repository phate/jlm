#include <stdlib.h>
#include <string.h>

static const char *
Create()
{
	return malloc(200);
}

static void
Copy(const char* p)
{
	char tmp[200];
	memcpy(tmp, p, 200);
}

int
main()
{
	const char * p = Create();
	Copy(p);
	return 0;
}
