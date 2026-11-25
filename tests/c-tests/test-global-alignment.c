// RUN: %jlc -Wall -Werror -O3 %s -o %t && %t

#include <stdint.h>
#include <assert.h>

int32_t __attribute__(( __aligned__(16) )) a;
int32_t __attribute__(( __aligned__(32) )) b;
int32_t __attribute__(( __aligned__(16) )) c;

int main()
{
    uintptr_t a_address = (uintptr_t) &a;
    uintptr_t b_address = (uintptr_t) &b;
    uintptr_t c_address = (uintptr_t) &c;

    assert(a_address % 16 == 0);
    assert(b_address % 32 == 0);
    assert(c_address % 16 == 0);
}
