#include <assert.h>

struct point {
	unsigned int x;
	unsigned int y;
};

struct point
add_points(struct point * p1, struct point * p2)
{
	struct point p;
	p.x = p1->x + p2->x;
	p.y = p1->y + p2->y;
	return p;
}

int
main()
{
	struct point p1 = {0, 0};
	struct point p2 = {1, 1};

	assert(p1.x == 0 && p1.y == 0);
	assert(p2.x == 1 && p2.y == 1);

	struct point p = add_points(&p1, &p2);
	assert(p.x == 1 && p.y == 1);

	return 0;
}
