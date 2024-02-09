#include <cilk/cilk.h>
#include <stdio.h>

int fib(int n) {
  if (n < 2)
    return n;
  int x, y;
  cilk_scope {
    x = cilk_spawn fib(n-1);
    y = fib(n-2);
  }
  return x+y;
}

int main() {
    int res = fib(10);
    printf("fib(10) = %d\n", res);
    return res;
}
