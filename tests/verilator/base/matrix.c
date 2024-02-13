//
// Created by david on 4/5/21.
//

#include <stdio.h>
#include <stdlib.h>

#define A(i, j) a[(i)*size + (j)]
#define B(i, j) b[(i)*size + (j)]
#define C_own(i, j) c[(i)*size + (j)]

void
reference(int * a, int * b, int * c, int size)
{
  for (int i = 0; i < size; i++)
    for (int j = 0; j < size; j++)
      for (int k = 0; k < size; k++)
        C_own(i, j) += A(i, k) * B(k, j);
}

void
run(int * a, int * b, int * c, int size)
{
  int i = 0;
  if (i < size)
  {
    do
    {
      int j = 0;
      do
      {
        int sum = C_own(i, j);
        int k = 0;
        do
        {
          sum += A(i, k) * B(k, j);
          k++;
        } while (k < size);
        C_own(i, j) = sum;
        j++;
      } while (j < size);
      i++;
    } while (i < size);
  }
}

void
init_dummy_data(int * a, int * b, int * c, int size)
{
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      A(i, j) = (int)(i + 1);
      B(i, j) = (int)(-i - 1);
      C_own(i, j) = 0;
    }
  }
}

void
print_matrix(int * a, int size)
{
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      printf("%d10", A(i, j));
    }
    printf("\n");
  }
}

int
check_matrix(int * a, int * b, int size)
{
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      if (A(i, j) != B(i, j))
      {
        return 1;
      }
    }
  }
  return 0;
}

int
main(int argc, char ** argv, char ** env)
{
  int size = 8;
  int * a = (int *)malloc(size * size * sizeof(int));
  int * b = (int *)malloc(size * size * sizeof(int));
  int * c = (int *)malloc(size * size * sizeof(int));
  int * r = (int *)calloc(size * size, sizeof(int));
  init_dummy_data(a, b, c, size);
  run(a, b, c, size);
  reference(a, b, r, size);
  print_matrix(c, size);
  return check_matrix(c, r, size);
}
