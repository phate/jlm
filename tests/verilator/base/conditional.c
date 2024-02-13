#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
run(int a, int b)
{
  if (a == b)
  {
    return -1;
  }
  else if (a > b)
  {
    return -2;
  }
  else if (a >= b)
  {
    return -3;
  }
  else if (b < a)
  {
    return -4;
  }
  else if (b <= a)
  {
    return -5;
  }

  return 1;
}

int
main(int argc, char ** argv)
{
  int result = run(4, 8);
  printf("Result: %i\n", result);

  // Check if correct result
  if (result == 1)
    return 0;

  return 1;
}
