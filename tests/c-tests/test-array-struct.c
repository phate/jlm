#include <assert.h>

struct point {
	unsigned int x;
	unsigned int y;
};

struct point points[100];

int
main()
{
	for (unsigned int n = 0; n < 100; n++) {
		points[n].x = 1;
		points[n].y = 2;
	}

	assert(points[60].x == 1 && points[60].y == 2);
}
