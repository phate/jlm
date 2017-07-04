#include <assert.h>

float test_fltadd(float x, float y){ return x + y; }
float test_fltsub(float x, float y){ return x - y; }
float test_fltmul(float x, float y){ return x * y; }
float test_fltdiv(float x, float y){ return x / y; }

double test_dbladd(double x, double y){ return x + y; }
double test_dblsub(double x, double y){ return x - y; }
double test_dblmul(double x, double y){ return x * y; }
double test_dbldiv(double x, double y){ return x / y; }

void
test_fltarithmetic()
{
	float x = 15.0;
	float y = 3.0;

	assert(test_fltadd(x, y) == 18.0);
	assert(test_fltdiv(x, y) == 5.0);
	assert(test_fltmul(x, y) == 45.0);
	assert(test_fltdiv(x, y) == 5.0);
}

void
test_dblarithmetic()
{
	double x = 15.0;
	double y = 3.0;

	assert(test_dbladd(x, y) == 18.0);
	assert(test_dbldiv(x, y) == 5.0);
	assert(test_dblmul(x, y) == 45.0);
	assert(test_dbldiv(x, y) == 5.0);
}

int
main()
{
	test_fltarithmetic();
	test_dblarithmetic();

	return 0;
}
