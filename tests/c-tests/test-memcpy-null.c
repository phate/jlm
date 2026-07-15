#include <string.h>

static char * buffer = NULL;
static char * buffer2 = NULL;

// Calling either of these functions will lead to undefined behavior,
// but the compiler should not crash

void
func1(int length)
{
    memcpy(buffer, buffer2, length);
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
