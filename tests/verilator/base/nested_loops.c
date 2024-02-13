#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
run(int a, int b)
{
  a = 1;
  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 5; j++)
    {
      a++;
    }
  }
  return a;
}

int
main(int argc, char ** argv)
{
  int result = run(1, 2);
  printf("Result: %i\n", result);

  // Check if correct result
  if (result == 26)
    return 0;

  return 1;
}
