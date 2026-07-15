#include <string.h>

static char * buffer = NULL;

// Calling either of these functions will lead to undefined behavior,
// but the compiler should not crash

void
func1(char* other, int length)
{
    memcpy(buffer, other, length);
}

void
func2(int value, int length)
{
    memset(buffer, value, length);
}

int main()
{
    return 0;
}
