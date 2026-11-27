// RUN: %jlc -Wall -Werror -J=PredicateCorrelation -J=LoopUnswitching %s -o %t && %t

int
add(int x)
{
  return x + 1;
}

int
mul(int x)
{
  return x * 2;
}

int
doit(int which, int x)
{
  int (*fp)(int) = add;
  int i;
  for (i = 0; i < 3; i++)
  {
    x = fp(x);
    if (which & 1)
      fp = mul;
    else
      fp = add;
    which >>= 1;
  }
  return x;
}

int
main(void)
{
  return doit(5, 1) == 5 ? 0 : 1;
}
