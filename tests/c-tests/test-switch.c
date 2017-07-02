#include <assert.h>

unsigned int
test_switch_fallthrough(unsigned int x)
{
	switch(x) {
		case 1:
		case 2:
		case 3:
			return 4;
		default:
			return 42;
	}
}

unsigned int
test_switch(unsigned int x)
{
	switch(x) {
		case 1: return 1;
		case 2: return 2;
		case 3: return 3;
		default: return 42;
	};
}

int
main()
{
	assert(test_switch(1) == 1);
	assert(test_switch(2) == 2);
	assert(test_switch(3) == 3);
	assert(test_switch(11) == 42);

	assert(test_switch_fallthrough(1) == 4);
	assert(test_switch_fallthrough(2) == 4);
	assert(test_switch_fallthrough(3) == 4);
	assert(test_switch_fallthrough(11) == 42);
	return 0;
}
