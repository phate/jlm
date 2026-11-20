// RUN: %jlc -Wall -Werror -O3 %s -o %t && %t

#include <assert.h>

static int c = 0;

int
get_next()
{
  return c++;
}

int
main()
{
  int status;

  while (1)
  {
    status = get_next();

    if (status == 5)
      break;
  }

  assert(status == 5);
  return 0;
}
