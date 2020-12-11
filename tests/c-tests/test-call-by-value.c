#include <assert.h>
#include <stdint.h>

struct test{
	uint64_t x;
	uint64_t y;
	uint64_t z;
};

static struct test g = {4, 5, 6};

uint32_t
sum(struct test t)
{
	t.x = 8;
	t.y = 2;
	t.z = 6;
	return t.x + t.y + t.z;
}

int
main()
{
	uint32_t r = sum(g);

	assert(r == 16);
	assert(g.x == 4);
	assert(g.y == 5);
	assert(g.z == 6);
}
