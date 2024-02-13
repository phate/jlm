#include "loop_array.h"
#include <stdio.h>
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

void
kernel(in_int_t n, in_int_t k, inout_int_t c[10])
{
  for (int i = 1; i < n; i++)
  {
    c[i] = k + c[i - 1];
  }
}

int
main(void)
{
  in_int_t k[AMOUNT_OF_TEST];
  in_int_t n[AMOUNT_OF_TEST];
  inout_int_t c[AMOUNT_OF_TEST][10];

  srand(13);
  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    k[i] = rand() % 10;
    n[i] = rand() % 10;
    for (int j = 0; j < 10; ++j)
    {
      c[i][j] = 0;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    kernel(n[i], k[i], c[i]);
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int j = 0; j < 10; ++j)
    {
      printf("%i ", c[i][j]);
    }
  }
}
