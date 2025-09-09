#include <stdio.h>

int main() {
  int i = 1;        // this will be an induction variable
  int sum = 0;      // this will NOT be an induction variable

  do {
    i = i + 2;    // induction variable: adds constant 2 each iteration
    sum = sum + i; // NOT induction variable: adds variable amount (i changes)
  } while (i < 10);

  return 0;
}
