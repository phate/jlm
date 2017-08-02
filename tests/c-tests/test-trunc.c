#include <assert.h>
#include <stdint.h>

uint32_t
tr(uint64_t x)
{
	return x;
}

int
main()
{
	int32_t x = tr(42);
	assert(x == 42);

	x = tr(0x00FFFFFFFFFFFFFF);
	assert(x == 0xFFFFFFFF);
	return 0;
}
