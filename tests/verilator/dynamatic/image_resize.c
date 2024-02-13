#include "image_resize.h"
#include <stdio.h>
#include <stdlib.h>

#define AMOUNT_OF_TEST 1

void
kernel(inout_int_t a[30][30], in_int_t c)
{
  for (int i = 0; i < 30; i++)
  {
    for (int j = 0; j < 30; j++)
    {
      a[i][j] = c - a[i][j];
    }
  }
}

int
main(void)
{
  inout_int_t a[AMOUNT_OF_TEST][30][30];
  in_int_t c[AMOUNT_OF_TEST];

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    c[i] = 1000;
    for (int y = 0; y < 30; ++y)
    {
      for (int x = 0; x < 30; ++x)
      {
        a[i][y][x] = rand() % 100;
      }
    }
  }

  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    kernel(a[i], c[i]);
  }
  for (int i = 0; i < AMOUNT_OF_TEST; ++i)
  {
    for (int y = 0; y < 30; ++y)
    {
      for (int x = 0; x < 30; ++x)
      {
        printf("%i ", a[i][y][x]);
      }
    }
  }
}
