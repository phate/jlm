#include <assert.h>

struct point {
	unsigned int x;
	unsigned int y;
};

struct point points[100];

static unsigned int array[] = {1, 2, 3};

static float array2[] = {1.0, 2.0, 3.0};

int
main()
{
	for (unsigned int n = 0; n < 100; n++) {
		points[n].x = 1;
		points[n].y = 2;
	}
	assert(points[60].x == 1 && points[60].y == 2);

	assert((array[0] + array[1] + array[2]) == 6);
	assert(array2[2] == 3.0);
}
