#include <assert.h>
#include <stdint.h>

int64_t
sext(int32_t x)
{
	return x;
}

int
main()
{
	int64_t x = sext(42);
	assert(x == 42);

	x = sext(0xFFFFFFFF);
	assert(x == 0xFFFFFFFFFFFFFFFF);

	return 0;
}
