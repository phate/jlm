#include <assert.h>

typedef unsigned long long bit64;
typedef unsigned int bit32;

typedef bit64 bit64x2 __attribute__((__vector_size__(16), __aligned__(16)));
typedef bit32 bit32x4 __attribute__((__vector_size__(16)));

typedef bit32 bit32x2 __attribute__((__vector_size__(8)));

// Converts two longs into four ints
bit32x4 splitLongsToInts(bit64x2 value)
{
    return (bit32x4) value;
}

// Combines two ints to a single long
bit64 combineIntsToLong(bit32x2 value)
{
    return (bit64) value;
}

int main() {
  bit64x2 longs = {0x1100000022, 0xAA000000BB};

  bit32x4 ints = splitLongsToInts(longs);

  // This assumes we are on a little-endian system
  assert(ints[0] == 0x22);
  assert(ints[1] == 0x11);
  assert(ints[2] == 0xBB);
  assert(ints[3] == 0xAA);

  bit32x2 intPair = {0x45, 0x78};
  bit64 combined = combineIntsToLong(intPair);

  assert(combined == 0x7800000045);
}
