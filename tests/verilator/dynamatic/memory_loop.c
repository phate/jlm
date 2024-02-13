#include "memory_loop.h"
#include <stdio.h>
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

void
kernel(inout_int_t x[1000], in_int_t y[1000])
{
  for (int i = 1; i < 1000; i++)
  {
    x[i] = x[i] * y[i] + x[0];
  }
}

int
main(void)
{
  inout_int_t x[AMOUNT_OF_TEST][1000];
  in_int_t y[AMOUNT_OF_TEST][1000];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int j = 0; j < 1000; ++j)
    {
      x[i][j] = rand() % 100;
      y[i][j] = rand() % 100;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    int i = 0;
    kernel(x[i], y[i]);
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int j = 0; j < 1000; ++j)
    {
      printf("%i ", x[i][j]);
    }
  }
}
