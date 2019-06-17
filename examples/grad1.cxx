#include <apex/autodiff_codegen.hxx>

std::array<double, 2> my_grad(double x, double y) {
	return apex::autodiff_grad("sq(x / y) +* sin(x)", { "x", "y" });
}

int main() {
  auto grad = my_grad(.3, .5);
  printf("%f %f\n", grad[0], grad[1]);
  return 0;
}