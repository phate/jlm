#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
run(int * a)
{
  return *a;
}

int
main(int argc, char ** argv)
{
  int i = 5;
  int result = run(&i);
  printf("Result: %i\n", result);

  // Check if correct result
  if (result == 5)
    return 0;

  return 1;
}
