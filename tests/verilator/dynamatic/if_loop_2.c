//------------------------------------------------------------------------
// If loop
//------------------------------------------------------------------------
#include "if_loop_2.h"
#include <stdio.h>
#include <stdlib.h>

int
kernel(in_int_t a[100], in_int_t n)
{
  int i;
  int tmp;
  int sum = 0;

  for (i = 0; i < n; i++)
  {
    tmp = a[i];
    if (tmp > 10)
    {
      sum = tmp + sum;
    }
  }
  return sum;
}

#define AMOUNT_OF_TEST 1

int
main(void)
{
  in_int_t a[AMOUNT_OF_TEST][100];
  in_int_t n[AMOUNT_OF_TEST];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    n[i] = 100;
    for (int j = 0; j < 100; ++j)
    {
      a[i][j] = rand() % 100;
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    printf("%i ", kernel(a[i], n[i]));
  }
}
