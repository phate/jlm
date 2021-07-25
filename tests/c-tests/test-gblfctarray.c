#include <assert.h>

int
c()
{
	return 0;
}

struct {
	int (*f)();
} isa[] = {
	{c}
};

int
main()
{
	int i = (*isa[0].f)();
	assert(i == 0);
}
