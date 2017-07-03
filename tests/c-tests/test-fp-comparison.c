#include <assert.h>

unsigned int test_flteq(float x, float y){ return x == y; }
unsigned int test_fltne(float x, float y){ return x != y; }
unsigned int test_fltgt(float x, float y){ return x > y; }
unsigned int test_fltge(float x, float y){ return x >= y; }
unsigned int test_fltlt(float x, float y){ return x < y; }
unsigned int test_fltle(float x, float y){ return x <= y; }

unsigned int test_dbleq(double x, double y){ return x == y; }
unsigned int test_dblne(double x, double y){ return x != y; }
unsigned int test_dblgt(double x, double y){ return x > y; }
unsigned int test_dblge(double x, double y){ return x >= y; }
unsigned int test_dbllt(double x, double y){ return x < y; }
unsigned int test_dblle(double x, double y){ return x <= y; }

void
test_fltcmp()
{
	float x = 4.5;
	float y = 0.2;

	assert(test_flteq(x, y) == 0);
	assert(test_flteq(x, x));

	assert(test_fltne(x, y));
	assert(test_fltne(x, x) == 0);

	assert(test_fltgt(x, y));
	assert(test_fltgt(y, x) == 0);

	assert(test_fltge(x, y));
	assert(test_fltge(y, x) == 0);
	assert(test_fltge(x, x));

	assert(test_fltlt(x, y) == 0);
	assert(test_fltlt(y, x));

	assert(test_fltle(x, y) == 0);
	assert(test_fltle(y, x));
	assert(test_fltle(x, x));
}

void
test_dblcmp()
{
	double x = 6.25;
	double y = 3.2;

	assert(test_dbleq(x, y) == 0);
	assert(test_dbleq(x, x));

	assert(test_dblne(x, y));
	assert(test_dblne(x, x) == 0);

	assert(test_dblgt(x, y));
	assert(test_dblgt(y, x) == 0);

	assert(test_dblge(x, y));
	assert(test_dblge(y, x) == 0);
	assert(test_dblge(x, x));

	assert(test_dbllt(x, y) == 0);
	assert(test_dbllt(y, x));

	assert(test_dblle(x, y) == 0);
	assert(test_dblle(y, x));
	assert(test_dblle(x, x));
}

int
main()
{
	test_fltcmp();
	test_dblcmp();

	return 0;
}
