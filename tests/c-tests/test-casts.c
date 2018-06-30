#include <assert.h>
#include <stdint.h>

int
fp2si(double d)
{
	return (int)d;
}

unsigned int
fp2ui(double d)
{
	return (unsigned int)d;
}

float
fptrunc(double d)
{
	return (float)d;
}

intptr_t
ptr2int(void * p)
{
	return (intptr_t)p;
}

void *
int2ptr(intptr_t p)
{
	return (void*)p;
}

int64_t
sext(int32_t x)
{
	return x;
}

float
sitofp(int x)
{
	return x;
}

double
ui2fp(unsigned i)
{
	return (double)(i);
}

uint32_t
tr(uint64_t x)
{
	return x;
}

int
main()
{
	assert(fp2si(-3.4) == -3);
	assert(fp2ui(3.4) == 3);
	assert(fptrunc(0.4) == (float)0.4);

	assert(sext(42) == 42);
	assert(sext(0xFFFFFFFF) == 0xFFFFFFFFFFFFFFFF);

	assert(tr(42) == 42);
	assert(tr(0x00FFFFFFFFFFFFFF) == 0xFFFFFFFF);

	assert(sitofp(-4) == -4.0);
	assert(ui2fp(3) == 3.0);

	return 0;
}
