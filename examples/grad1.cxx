#include <apex/autodiff_codegen.hxx>

struct terms_t {
  double x;
  double y;
};

terms_t my_grad(terms_t input) {
  return apex::autodiff_grad("sq(x / y) * sin(x * y)", input);
}

int main() {
  terms_t grad = my_grad( { .3, .5 } );
  printf("%f %f\n", grad.x, grad.y);
  return 0;
}