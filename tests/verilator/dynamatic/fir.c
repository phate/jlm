//------------------------------------------------------------------------
// FIR
//------------------------------------------------------------------------
#include "fir.h"
#include <stdio.h>
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

int
kernel(in_int_t d_i[1000], in_int_t idx[1000])
{
  int i;
  int tmp = 0;

  for (i = 0; i < 1000; i++)
  {
    tmp += idx[i] * d_i[999 - i];
  }

  return tmp;
}

int
main(void)
{
  in_int_t d_i[AMOUNT_OF_TEST][1000];
  in_int_t idx[AMOUNT_OF_TEST][1000];
  inout_int_t out[AMOUNT_OF_TEST][1000];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int j = 0; j < 1000; ++j)
    {
      d_i[0][j] = rand() % 100;
      idx[0][j] = rand() % 100;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    printf("%i ", kernel(d_i[0], idx[0]));
  }
}
