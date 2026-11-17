/* Function pointer whose value alternates between two functions inside a loop.
 *
 * This is the C-level shape that produces a function-typed SSA phi in the
 * backend: PredicateCorrelation + LoopUnswitching route the two callee values
 * through a loop-carried selection, so an SsaPhiOperation over FunctionType
 * reaches IpGraphToLlvmConverter::convert_phi. Functions are pointers in LLVM
 * IR, so that phi must be converted as a pointer-typed phi.
 *
 * The plain -O3 pipeline used by the c-test harness avoids the pattern, but
 * it can be observed with:
 *
 *   jlc -J=PredicateCorrelation -J=LoopUnswitching -o test-fnptr-loop test-fnptr-loop.c
 *
 * The unit test IpGraphToLlvmConverterTests.SsaPhiWithFunctionType covers the
 * conversion deterministically.
 */

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
