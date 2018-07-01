#include <assert.h>

typedef struct {
	double real;
	double imag;
} complex;

complex
create(double real, double imag)
{
	complex cc = {real, imag};
	return cc;
}

int
main()
{
	complex cc = create(4.0, 3.0);

	assert(cc.real == 4.0 && cc.imag == 3.0);

	return 0;
}
