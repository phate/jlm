#include <assert.h>
#include <stdint.h>

typedef int64_t int64x2 __attribute__((__vector_size__(16), __aligned__(16)));
typedef int32_t int32x4 __attribute__((__vector_size__(16)));
typedef int32_t int32x2 __attribute__((__vector_size__(8)));

// Converts two longs into four ints
int32x4 splitLongsToInts(int64x2 value)
{
    return (int32x4) value;
}

// Combines two ints to a single long
int64_t combineIntsToLong(int32x2 value)
{
    return (int64_t) value;
}

int main() {
  int64x2 longs = {0x1100000022, 0xAA000000BB};

  int32x4 ints = splitLongsToInts(longs);

  // This assumes we are on a little-endian system
  assert(ints[0] == 0x22);
  assert(ints[1] == 0x11);
  assert(ints[2] == 0xBB);
  assert(ints[3] == 0xAA);

  int32x2 intPair = {0x45, 0x78};
  int64_t combined = combineIntsToLong(intPair);

  assert(combined == 0x7800000045);
}
