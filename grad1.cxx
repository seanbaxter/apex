#include <apex/autodiff.hxx>
#include <array>
#include <cmath>

BEGIN_APEX_NAMESPACE

// STL doesn't include a sq function, but it's really useful, because it 
// memoizes the argument, so we don't have to evaluate its subexpression twice
// to take its square.
inline double sq(double x) {
  return x * x;
}

@macro auto autodiff_expr(const ad_t* ad) {
  @meta+ if(const auto* tape = ad->as<ad_tape_t>()) {
    @emit return tape_values[tape->index];

  } else if(const auto* literal = ad->as<ad_literal_t>()) {
    @emit return literal->x;

  } else if(const auto* unary = ad->as<ad_unary_t>()) {
    @emit return @op(
      unary->op, 
      autodiff_expr(unary->a.get())
    );

  } else if(const auto* binary = ad->as<ad_binary_t>()) {
    @emit return @op(
      binary->op, 
      autodiff_expr(binary->a.get()), 
      autodiff_expr(binary->b.get())
    );

  } else if(const auto* func = ad->as<ad_func_t>()) {
    /*
    @emit return @expression(func->f)(
      // Evaluate and expand the arguments parameter pack.
      autodiff_expr(func->args[__integer_pack(func->args.size())].get())...
    );
    */
    @emit return @expression(func->f)(autodiff_expr(func->args[0].get()));
  }
}

@macro void autodiff_grad(int index, int parent) {
  @meta if(index < num_vars) {
    // We've hit a terminal, which corresponds to an independent variable.
    // Increment the gradient array by the coefficient at parent.
    grad[index] += coef[parent];

  } else {
    // We're in a subexpression. Evaluate each of the child nodes.
    @meta for(const auto& g : ad_builder.tape[index].grads) {
      // Evaluate the coefficient into the stack.
      coef[index] = coef[parent] * autodiff_expr(g.coef.get());
      @macro autodiff_grad(g.index, index);
    }
  }
}

template<typename... args_t>
@meta std::array<double, sizeof...(args_t)> autodiff_eval(
  @meta const ad_builder_t& ad_builder, args_t... args) {

  @meta size_t num_vars = ad_builder.var_names.size();
  @meta size_t num_items = ad_builder.tape.size();

  // Compute the values for the whole tape. This is the forward-mode pass. 
  // It propagates values from the terminals (independent variables) through
  // the subexpressions and up to the root of the function.

  // Copy the values of the independent variables into the tape.
  double tape_values[num_items] { args... };

  // Evaluate the subexpressions.
  @meta for(size_t i = num_vars; i < num_items; ++i)
    tape_values[i] = autodiff_expr(ad_builder.tape[i].val.get());

  // Evaluate the gradients. This is a top-down reverse-mode traversal of 
  // the autodiff DAG. The partial derivatives are parsed along edges, starting
  // from the root and towards each terminal. When a terminal is visited, the
  // corresponding component of the gradient is incremented by the product of
  // all the partial derivatives from the root of the DAG down to that 
  // terminal.

  double coef[num_items];
  std::array<double, num_vars> grad { };

  // Visit each child of the root node.
  @meta int root = num_items - 1;
  @meta for(const auto& g : ad_builder.tape[root].grads) {
    // Evaluate the coefficient into the stack.
    coef[root] = autodiff_expr(g.coef.get());

    // Recurse on the child.
    @macro autodiff_grad(g.index, root);
  }

  return std::move(grad);
}

@macro auto solve_autodiff(std::string __fmt, 
  std::vector<std::string> __var_names) {

  // Construct the autodiff builder.
  @meta apex::ad_builder_t __ad_builder;
  @meta __ad_builder.process(__fmt, __var_names);

  @meta std::vector<std::string> varnames = __var_names;

  return autodiff_eval(
    __ad_builder, 
    @expression(varnames[__integer_pack(__var_names.size())])...
  );
}

END_APEX_NAMESPACE

int main() {
  double x = .3;
  auto grad = apex::solve_autodiff("sq(x) * sin(x)", { "x" });
  printf("%f\n", grad[0]);
  return 0;
}