#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
run(int a, int b)
{
  int c = ((a + b) * b - a) / b % a;
  return c;
}

int
main(int argc, char ** argv)
{
  int result = run(4, 8);
  printf("Result: %i\n", result);

  // Check if correct result
  if (result == 3)
    return 0;

  return 1;
}
