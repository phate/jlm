//------------------------------------------------------------------------
// Pivot
//------------------------------------------------------------------------
#include "pivot.h"
#include <stdio.h>
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

void
kernel(int x[1000], int a[1000], int n, int k)
{
  int i;

  for (i = k + 1; i <= n; ++i)
  {
    x[k] = x[k] - a[i] * x[i];
  }
}

int
main(void)
{
  inout_int_t x[AMOUNT_OF_TEST][1000];
  in_int_t a[AMOUNT_OF_TEST][1000];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int j = 0; j < 1000; ++j)
    {
      x[i][j] = rand() % 100;
      a[i][j] = rand() % 100;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    kernel(x[i], a[i], 100, 2);
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int j = 0; j < 1000; ++j)
    {
      printf("%i ", x[i][j]);
    }
  }
}
