#include <apex/autodiff_codegen.hxx>

int main() {
  double x = .3;
  double y = .5;
  auto grad = apex::solve_autodiff("sq(x / y) * pow(x,y) * sin(x)", { "x", "y" });
  printf("%f %f\n", grad[0], grad[1]);
  return 0;
}