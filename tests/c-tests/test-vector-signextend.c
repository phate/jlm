#include <assert.h>
#include <stdint.h>

typedef uint32_t uint32x4 __attribute__((__vector_size__(16), __aligned__(16)));
typedef uint16_t uint16x4 __attribute__((__vector_size__(8), __aligned__(8)));

uint32x4
equals(uint32x4 a, uint32x4 b)
{
  // Unlike scalar == where equality gives the result value 1,
  // the SIMD == fills the lane with all 1 bits when operands are equal.
  return a == b;
}

int
main()
{
  uint32x4 a = {5, 20, 8, 42};
  uint32x4 b = {8, 20, 9, 42};

  uint32x4 equality = equals(a, b);

  assert(equality[0] == 0);
  assert(equality[1] == 0xFFFFFFFF);
  assert(equality[2] == 0);
  assert(equality[3] == 0xFFFFFFFF);

  return 0;
}
