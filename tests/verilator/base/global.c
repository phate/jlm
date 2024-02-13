#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int C[32];
int * t;

void
run()
{
  for (int i = 0; i < 32; i++)
  {
    C[i] = i * i + *t;
  }
}

int
main(int argc, char ** argv)
{
  int a = 0;
  t = &a;
  run();

  /*
  for (int i =0; i<32; i++)
      printf("i=%i\ti*i+a=%i\t%i\n", i, i*i+a, C[i]);
  */

  int i = 0;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;

  a = 10;
  run();

  i = 0;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;
  assert(C[i] == i * i + a);
  i++;

  return 0;
}
